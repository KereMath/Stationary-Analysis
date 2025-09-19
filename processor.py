"""
Time Series Stationarity Classification - Data Processing Module
Handles large-scale CSV processing with memory efficiency
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import gc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from config import DATA_PATH, PROCESSED_DATA_DIR, CHUNK_SIZE, FILES_PER_FOLDER_LIMIT, LABEL_MAP

warnings.filterwarnings('ignore')

# process_single_file metodunu sınıf dışına alıp, daha kolay map'lenebilir hale getireceğiz.
# Ancak sınıf içindeki helper metodları kullandığı için, sınıfın bir kopyasını da almalı.
# Daha temiz bir çözüm için, wrapper metodu kullanalım.

class TimeSeriesDataProcessor:
    """Efficient processor for large-scale time series data"""
    
    def __init__(self, base_path: str, chunk_size: int = 10000):
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.file_paths = {'stationary': [], 'non_stationary': []}
        
    def scan_directories(self):
        print("Scanning directories for CSV files...")
        
        for folder in self.base_path.iterdir():
            if folder.name.startswith('__') or folder.name.startswith('.'):
                continue

            if folder.is_dir():
                label_name = 'stationary' if folder.name.lower() == 'stationary' else 'non_stationary'
                
                csv_files = list(folder.rglob('*.csv'))
                valid_files = [f for f in csv_files if 'metadata' not in f.name.lower()]

                if FILES_PER_FOLDER_LIMIT is not None:
                    print(f"  -> '{folder.name}' klasöründen {FILES_PER_FOLDER_LIMIT} dosya alınıyor...")
                    valid_files = valid_files[:FILES_PER_FOLDER_LIMIT]

                self.file_paths[label_name].extend(valid_files)
                
        print(f"\nFound {len(self.file_paths['stationary'])} stationary files for processing.")
        print(f"Found {len(self.file_paths['non_stationary'])} non-stationary files for processing.")
        return self.file_paths
    
    # ... Diğer tüm _calculate, _rolling, _autocorrelation vb. helper fonksiyonları burada AYNEN kalacak ...
    def extract_features_from_chunk(self, data: np.ndarray) -> Optional[Dict]:
        if len(data) < 2: return None
        features = {}
        try:
            features['mean'] = np.mean(data); features['std'] = np.std(data); features['var'] = np.var(data)
            features['min'] = np.min(data); features['max'] = np.max(data); features['range'] = features['max'] - features['min']
            features['q25'] = np.percentile(data, 25); features['median'] = np.median(data); features['q75'] = np.percentile(data, 75)
            features['iqr'] = features['q75'] - features['q25']; features['skewness'] = self._calculate_skewness(data)
            features['kurtosis'] = self._calculate_kurtosis(data); features['cv'] = features['std'] / (features['mean'] + 1e-10)
            diff1 = np.diff(data); features['diff1_mean'] = np.mean(diff1); features['diff1_std'] = np.std(diff1); features['diff1_var'] = np.var(diff1)
            if len(diff1) > 1:
                diff2 = np.diff(diff1); features['diff2_mean'] = np.mean(diff2); features['diff2_std'] = np.std(diff2)
            else:
                features['diff2_mean'] = 0; features['diff2_std'] = 0
            window_size = max(2, len(data) // 10)
            if window_size < len(data):
                rolling_means = self._rolling_window_stat(data, window_size, np.mean); rolling_stds = self._rolling_window_stat(data, window_size, np.std)
                features['rolling_mean_std'] = np.std(rolling_means); features['rolling_std_mean'] = np.mean(rolling_stds); features['rolling_std_std'] = np.std(rolling_stds)
            else:
                features['rolling_mean_std'] = 0; features['rolling_std_mean'] = features['std']; features['rolling_std_std'] = 0
            features['autocorr_lag1'] = self._autocorrelation(data, 1); features['autocorr_lag10'] = self._autocorrelation(data, min(10, len(data)-1))
            features['num_peaks'] = self._count_peaks(data); features['zero_crossing_rate'] = self._zero_crossing_rate(data - np.mean(data))
        except Exception: return None
        return features
    def _calculate_skewness(self, data: np.ndarray) -> float:
        n = len(data); mean = np.mean(data); std = np.std(data)
        if std == 0 or n < 3: return 0
        return (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        n = len(data); mean = np.mean(data); std = np.std(data)
        if std == 0 or n < 4: return 0
        return (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n-1)**2 / ((n-2) * (n-3)))
    def _rolling_window_stat(self, data: np.ndarray, window: int, func) -> np.ndarray:
        return np.array([func(data[i:i+window]) for i in range(len(data) - window + 1)])
    def _autocorrelation(self, data: np.ndarray, lag: int) -> float:
        if lag >= len(data) or lag < 1: return 0
        n = len(data); mean = np.mean(data); c0 = np.sum((data - mean) ** 2) / n
        if c0 == 0: return 0
        ck = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / n
        return ck / c0
    def _count_peaks(self, data: np.ndarray) -> int:
        if len(data) < 3: return 0
        return sum(1 for i in range(1, len(data) - 1) if data[i] > data[i-1] and data[i] > data[i+1])
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        if len(data) < 2: return 0
        return np.sum(np.diff(np.sign(data)) != 0) / (len(data) - 1)
    def _aggregate_chunk_features(self, chunks_features: List[Dict]) -> Dict:
        if not chunks_features: return {}
        if len(chunks_features) == 1: return chunks_features[0]
        aggregated = {}
        feature_names = chunks_features[0].keys()
        for feature in feature_names:
            values = [chunk[feature] for chunk in chunks_features]
            aggregated[f'{feature}_mean'] = np.mean(values)
            aggregated[f'{feature}_std'] = np.std(values)
        return aggregated

    # Bu iki fonksiyonu process_files_parallel'in düzgün çalışması için ekliyoruz.
    def process_single_file(self, file_path: Path, label: int) -> Optional[Tuple[np.ndarray, int]]:
        try:
            chunks_features = []
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, usecols=['data']):
                if 'data' not in chunk.columns: continue
                data_values = chunk['data'].dropna().values
                if len(data_values) > 1:
                    features = self.extract_features_from_chunk(data_values)
                    if features: chunks_features.append(features)
            
            if chunks_features:
                aggregated_features = self._aggregate_chunk_features(chunks_features)
                feature_vector = np.array(list(aggregated_features.values()))
                return feature_vector, label
        except Exception:
            return None
        return None

    @staticmethod
    def _process_file_static(args):
        # Bu statik metot, ProcessPoolExecutor.map tarafından çağrılabilir olacak.
        # Gerekli tüm bilgileri 'args' ile alır.
        file_path, label, chunk_size_instance = args
        
        # Sınıfın geçici bir örneğini oluşturup metodları kullanalım
        temp_processor = TimeSeriesDataProcessor(base_path='', chunk_size=chunk_size_instance)
        return temp_processor.process_single_file(file_path, label)

    def process_files_parallel(self) -> Tuple[np.ndarray, np.ndarray]:
        stationary_files = self.file_paths['stationary']
        non_stationary_files = self.file_paths['non_stationary']
        all_files_tuples = [(f, LABEL_MAP['stationary']) for f in stationary_files] + \
                           [(f, LABEL_MAP['non_stationary']) for f in non_stationary_files]
        
        if not all_files_tuples: return np.array([]), np.array([])
        
        print(f"\nProcessing {len(all_files_tuples)} files in parallel (Optimized)...")
        
        # --- OPTİMİZASYONLAR BURADA ---
        n_workers = 4  # 1. Optimizasyon: Çalışan sayısını sabitle
        chunk_size_for_map = 100 # 2. Optimizasyon: Görevleri 100'lük partiler halinde dağıt

        # Statik metoda göndereceğimiz argüman listesini hazırlayalım
        task_args = [(fp, lbl, self.chunk_size) for fp, lbl in all_files_tuples]
        
        X, y = [], []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # executor.map, görevleri partiler halinde (chunksize) gönderir, bu da verimliliği artırır.
            results = list(tqdm(
                executor.map(self._process_file_static, task_args, chunksize=chunk_size_for_map),
                total=len(all_files_tuples),
                desc="Processing Files"
            ))

        for result in results:
            if result:
                feature_vector, label = result
                X.append(feature_vector)
                y.append(label)
        
        if X:
            lengths = {len(row) for row in X}
            if len(lengths) > 1:
                max_cols = max(lengths)
                print(f"Feature vectors have inconsistent lengths. Padding to {max_cols}.")
                X = [np.pad(row, (0, max_cols - len(row)), 'constant') if len(row) < max_cols else row for row in X]

        return np.array(X), np.array(y)

    def save_processed_data(self, X: np.ndarray, y: np.ndarray, output_dir: str):
        # Bu fonksiyon aynı kalabilir
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'features.npy'), X)
        np.save(os.path.join(output_dir, 'labels.npy'), y)
        print(f"Saved processed data to {output_dir}")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

def run_processing():
    print("--- Veri İşleme Aşaması Başladı ---")
    processor = TimeSeriesDataProcessor(base_path=DATA_PATH, chunk_size=CHUNK_SIZE)
    processor.scan_directories()
    X, y = processor.process_files_parallel()
    if X.shape[0] > 0:
        processor.save_processed_data(X, y, output_dir=str(PROCESSED_DATA_DIR))
        print("--- Veri İşleme Aşaması Tamamlandı ---")
    else:
        print("İşlenecek veri bulunamadı.")

if __name__ == "__main__":
    run_processing()