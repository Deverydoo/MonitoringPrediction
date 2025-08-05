#!/usr/bin/env python3
"""
data_converter.py - Efficient Data Format Converter
Convert large JSON datasets to efficient formats (Parquet, HDF5)
Can be used as both importable module and command-line tool
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional imports for different storage formats
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import tables
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

# Import project modules
try:
    from common_utils import log_message
except ImportError:
    def log_message(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


class DataFormatConverter:
    """Efficient converter for large time series datasets."""
    
    def __init__(self, chunk_size: int = 50000):
        """
        Initialize converter.
        
        Args:
            chunk_size: Number of records to process at once
        """
        self.chunk_size = chunk_size
        self.supported_formats = self._get_supported_formats()
    
    def _get_supported_formats(self) -> Dict[str, bool]:
        """Get supported output formats."""
        return {
            'parquet': PARQUET_AVAILABLE,
            'hdf5': HDF5_AVAILABLE,
            'csv': True,  # Always available
            'feather': PARQUET_AVAILABLE,  # Uses pyarrow
        }
    
    def convert_json_dataset(self, 
                           input_path: Union[str, Path],
                           output_format: str = 'parquet',
                           output_path: Optional[Union[str, Path]] = None,
                           compression: str = 'auto') -> Optional[Path]:
        """
        Convert JSON dataset to efficient format.
        
        Args:
            input_path: Path to input JSON file
            output_format: Target format ('parquet', 'hdf5', 'csv', 'feather')
            output_path: Optional output path (auto-generated if None)
            compression: Compression method ('auto', 'snappy', 'gzip', etc.)
            
        Returns:
            Path to converted file, or None if failed
        """
        input_file = Path(input_path)
        if not input_file.exists():
            log_message(f"❌ Input file not found: {input_path}")
            return None
        
        if output_format not in self.supported_formats:
            log_message(f"❌ Unsupported format: {output_format}")
            return None
        
        if not self.supported_formats[output_format]:
            log_message(f"❌ Required dependencies not available for {output_format}")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_file.with_suffix(f'.{output_format}')
        else:
            output_path = Path(output_path)
        
        log_message(f"🔄 Converting {input_file.name} to {output_format.upper()}")
        log_message(f"📁 Output: {output_path}")
        
        start_time = time.time()
        
        try:
            # Load and parse JSON
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'training_samples' not in data:
                log_message("❌ Invalid JSON format - missing 'training_samples'")
                return None
            
            samples = data['training_samples']
            metadata = data.get('metadata', {})
            
            log_message(f"📊 Processing {len(samples):,} samples in chunks of {self.chunk_size:,}")
            
            # Convert samples to DataFrame in chunks
            df_chunks = []
            
            for i in tqdm(range(0, len(samples), self.chunk_size), desc="Processing chunks"):
                chunk_samples = samples[i:i + self.chunk_size]
                chunk_df = self._samples_to_dataframe(chunk_samples)
                df_chunks.append(chunk_df)
            
            # Combine chunks
            log_message("🔗 Combining chunks...")
            df = pd.concat(df_chunks, ignore_index=True)
            
            # Optimize data types
            df = self._optimize_dtypes(df)
            
            log_message(f"✅ DataFrame created: {df.shape}")
            log_message(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Save in requested format
            success = self._save_dataframe(df, output_path, output_format, compression)
            
            if success:
                # Save metadata separately
                metadata_path = output_path.with_suffix('.metadata.json')
                self._save_metadata(metadata, metadata_path, df)
                
                # Show conversion results
                elapsed_time = time.time() - start_time
                input_size_mb = input_file.stat().st_size / (1024 * 1024)
                output_size_mb = output_path.stat().st_size / (1024 * 1024)
                compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0
                
                log_message(f"🎉 Conversion completed in {elapsed_time:.1f} seconds")
                log_message(f"📊 Size: {input_size_mb:.1f} MB → {output_size_mb:.1f} MB")
                log_message(f"🗜️  Compression: {compression_ratio:.1f}x")
                
                return output_path
            else:
                return None
                
        except Exception as e:
            log_message(f"❌ Conversion failed: {e}")
            return None
    
    def _samples_to_dataframe(self, samples: List[Dict]) -> pd.DataFrame:
        """Convert sample list to DataFrame efficiently."""
        records = []
        
        for sample in samples:
            record = {
                'timestamp': sample.get('timestamp'),
                'server_name': sample.get('server_name', 'unknown'),
                'status': sample.get('status', 'normal'),
                'timeframe': sample.get('timeframe', 'unknown'),
                'severity': sample.get('severity', 'low')
            }
            
            # Flatten metrics
            metrics = sample.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                record[metric_name] = metric_value
            
            # Add server profile info if available
            server_profile = sample.get('server_profile', {})
            for profile_key, profile_value in server_profile.items():
                record[f'profile_{profile_key}'] = profile_value
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for storage."""
        df = df.copy()
        
        log_message("🔧 Optimizing data types...")
        
        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'timestamp':  # Don't categorize timestamps
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
                    log_message(f"   📂 {col}: object → category")
        
        # Optimize integer types
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:  # Signed integers
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float types
        for col in df.select_dtypes(include=['float64']).columns:
            # Try float32 if precision loss is acceptable
            float32_version = df[col].astype('float32')
            if np.allclose(df[col].dropna(), float32_version.dropna(), rtol=1e-6, equal_nan=True):
                df[col] = float32_version
                log_message(f"   📊 {col}: float64 → float32")
        
        return df
    
    def _save_dataframe(self, 
                       df: pd.DataFrame, 
                       output_path: Path, 
                       output_format: str, 
                       compression: str) -> bool:
        """Save DataFrame in specified format."""
        try:
            if output_format == 'parquet':
                return self._save_parquet(df, output_path, compression)
            elif output_format == 'hdf5':
                return self._save_hdf5(df, output_path, compression)
            elif output_format == 'csv':
                return self._save_csv(df, output_path, compression)
            elif output_format == 'feather':
                return self._save_feather(df, output_path, compression)
            else:
                log_message(f"❌ Unsupported format: {output_format}")
                return False
                
        except Exception as e:
            log_message(f"❌ Failed to save {output_format}: {e}")
            return False
    
    def _save_parquet(self, df: pd.DataFrame, output_path: Path, compression: str) -> bool:
        """Save as Parquet format."""
        if compression == 'auto':
            compression = 'snappy'  # Good balance of speed and compression
        
        log_message(f"💾 Saving as Parquet with {compression} compression...")
        
        # Create PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Write with optimization
        pq.write_table(
            table,
            output_path,
            compression=compression,
            use_dictionary=True,  # Compress string columns
            row_group_size=50000,  # Optimize for reading chunks
            write_statistics=True  # Enable column statistics
        )
        
        return True
    
    def _save_hdf5(self, df: pd.DataFrame, output_path: Path, compression: str) -> bool:
        """Save as HDF5 format."""
        if compression == 'auto':
            compression = 'zlib'
        
        log_message(f"💾 Saving as HDF5 with {compression} compression...")
        
        # Save with compression
        with pd.HDFStore(
            output_path, 
            mode='w', 
            complevel=6 if compression else 0,
            complib=compression if compression != 'auto' else 'zlib'
        ) as store:
            store.put('metrics', df, format='table', data_columns=True)
        
        return True
    
    def _save_csv(self, df: pd.DataFrame, output_path: Path, compression: str) -> bool:
        """Save as CSV format."""
        if compression == 'auto':
            compression = 'gzip'
        
        log_message(f"💾 Saving as CSV with {compression} compression...")
        
        # Adjust output path for compression
        if compression and compression != 'none':
            if not output_path.suffix == f'.{compression}':
                output_path = output_path.with_suffix(f'{output_path.suffix}.{compression}')
        
        df.to_csv(output_path, index=False, compression=compression)
        return True
    
    def _save_feather(self, df: pd.DataFrame, output_path: Path, compression: str) -> bool:
        """Save as Feather format."""
        if compression == 'auto':
            compression = 'zstd'
        
        log_message(f"💾 Saving as Feather with {compression} compression...")
        
        df.to_feather(output_path, compression=compression)
        return True
    
    def _save_metadata(self, metadata: Dict, metadata_path: Path, df: pd.DataFrame):
        """Save metadata with dataset statistics."""
        enhanced_metadata = metadata.copy()
        
        # Add conversion statistics
        enhanced_metadata.update({
            'converted_at': datetime.now().isoformat(),
            'converted_format': metadata_path.parent.name,
            'dataframe_shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'column_dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        log_message(f"📝 Metadata saved: {metadata_path}")
    
    def compare_formats(self, input_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """Compare different output formats for the same dataset."""
        input_file = Path(input_path)
        if not input_file.exists():
            log_message(f"❌ Input file not found: {input_path}")
            return {}
        
        log_message("📊 Comparing output formats...")
        
        results = {}
        temp_dir = input_file.parent / 'format_comparison'
        temp_dir.mkdir(exist_ok=True)
        
        formats_to_test = ['parquet', 'hdf5', 'csv', 'feather']
        
        for fmt in formats_to_test:
            if not self.supported_formats.get(fmt, False):
                continue
            
            log_message(f"🧪 Testing {fmt.upper()} format...")
            
            start_time = time.time()
            output_path = self.convert_json_dataset(
                input_path, 
                fmt, 
                temp_dir / f'test_dataset.{fmt}',
                compression='auto'
            )
            conversion_time = time.time() - start_time
            
            if output_path and output_path.exists():
                # Test read performance
                read_start = time.time()
                if fmt == 'parquet':
                    test_df = pd.read_parquet(output_path)
                elif fmt == 'hdf5':
                    test_df = pd.read_hdf(output_path, 'metrics')
                elif fmt == 'csv':
                    test_df = pd.read_csv(output_path)
                elif fmt == 'feather':
                    test_df = pd.read_feather(output_path)
                read_time = time.time() - read_start
                
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                
                results[fmt] = {
                    'file_size_mb': file_size_mb,
                    'conversion_time_sec': conversion_time,
                    'read_time_sec': read_time,
                    'rows': len(test_df),
                    'columns': len(test_df.columns),
                    'success': True
                }
                
                # Cleanup test file
                output_path.unlink()
                if output_path.with_suffix('.metadata.json').exists():
                    output_path.with_suffix('.metadata.json').unlink()
            else:
                results[fmt] = {'success': False}
        
        # Cleanup temp directory
        if temp_dir.exists() and not list(temp_dir.iterdir()):
            temp_dir.rmdir()
        
        # Display comparison
        log_message("\n📊 FORMAT COMPARISON RESULTS:")
        log_message("=" * 60)
        
        for fmt, stats in results.items():
            if stats.get('success'):
                log_message(f"{fmt.upper():8s}: {stats['file_size_mb']:6.1f} MB | "
                          f"Convert: {stats['conversion_time_sec']:5.1f}s | "
                          f"Read: {stats['read_time_sec']:5.1f}s")
            else:
                log_message(f"{fmt.upper():8s}: FAILED")
        
        return results


# Module interface functions
def convert_to_parquet(json_path: str, parquet_path: str = None) -> Optional[str]:
    """Convert JSON to Parquet format."""
    converter = DataFormatConverter()
    result = converter.convert_json_dataset(json_path, 'parquet', parquet_path)
    return str(result) if result else None


def convert_to_hdf5(json_path: str, hdf5_path: str = None) -> Optional[str]:
    """Convert JSON to HDF5 format."""
    converter = DataFormatConverter()
    result = converter.convert_json_dataset(json_path, 'hdf5', hdf5_path)
    return str(result) if result else None


def get_best_format_recommendation(json_path: str) -> str:
    """Get recommended format based on dataset characteristics."""
    json_file = Path(json_path)
    file_size_mb = json_file.stat().st_size / (1024 * 1024)
    
    if file_size_mb < 50:
        return "csv"  # Small files, CSV is fine
    elif file_size_mb < 500:
        return "parquet"  # Medium files, Parquet is best
    else:
        return "hdf5"  # Large files, HDF5 might be better for sequential access


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert large JSON datasets to efficient formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to Parquet (default)
  python data_converter.py input.json
  
  # Convert to HDF5
  python data_converter.py input.json --format hdf5
  
  # Specify output path
  python data_converter.py input.json --output output.parquet
  
  # Compare all formats
  python data_converter.py input.json --compare
  
  # Show format recommendations
  python data_converter.py input.json --recommend
        """
    )
    
    parser.add_argument(
        'input', type=str,
        help='Input JSON file path'
    )
    parser.add_argument(
        '--format', '-f', 
        choices=['parquet', 'hdf5', 'csv', 'feather'],
        default='parquet',
        help='Output format (default: parquet)'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output file path (auto-generated if not specified)'
    )
    parser.add_argument(
        '--compression', '-c',
        choices=['auto', 'snappy', 'gzip', 'zlib', 'zstd', 'lz4', 'none'],
        default='auto',
        help='Compression method (default: auto)'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=50000,
        help='Processing chunk size (default: 50000)'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare all available formats'
    )
    parser.add_argument(
        '--recommend', action='store_true',
        help='Show format recommendation'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        log_message(f"❌ Input file not found: {args.input}")
        return 1
    
    # Create converter
    converter = DataFormatConverter(chunk_size=args.chunk_size)
    
    # Show supported formats
    if args.verbose:
        log_message("🔧 Supported formats:")
        for fmt, available in converter.supported_formats.items():
            status = "✅" if available else "❌"
            log_message(f"   {status} {fmt.upper()}")
    
    try:
        if args.recommend:
            # Show recommendation
            recommendation = get_best_format_recommendation(args.input)
            file_size_mb = input_path.stat().st_size / (1024 * 1024)
            
            log_message(f"💡 RECOMMENDATION for {file_size_mb:.1f} MB dataset:")
            log_message(f"   Best format: {recommendation.upper()}")
            
            if recommendation == 'parquet':
                log_message("   Reasons: Excellent compression, fast analytics, wide support")
            elif recommendation == 'hdf5':
                log_message("   Reasons: Great for large datasets, efficient for time series")
            elif recommendation == 'csv':
                log_message("   Reasons: Small dataset, universal compatibility")
            
            return 0
        
        elif args.compare:
            # Compare formats
            results = converter.compare_formats(args.input)
            
            if results:
                # Find best format
                successful_results = {k: v for k, v in results.items() if v.get('success')}
                if successful_results:
                    best_size = min(successful_results.values(), key=lambda x: x['file_size_mb'])
                    best_speed = min(successful_results.values(), key=lambda x: x['conversion_time_sec'])
                    
                    log_message(f"\n🏆 RECOMMENDATIONS:")
                    log_message(f"   Smallest size: {[k for k, v in successful_results.items() if v == best_size][0].upper()}")
                    log_message(f"   Fastest conversion: {[k for k, v in successful_results.items() if v == best_speed][0].upper()}")
            
            return 0
            
        else:
            # Convert to specified format
            output_path = converter.convert_json_dataset(
                args.input,
                args.format,
                args.output,
                args.compression
            )
            
            if output_path:
                log_message(f"✅ Conversion successful: {output_path}")
                
                # Show usage instructions
                log_message(f"\n💡 To use this dataset:")
                if args.format == 'parquet':
                    log_message(f"   df = pd.read_parquet('{output_path}')")
                elif args.format == 'hdf5':
                    log_message(f"   df = pd.read_hdf('{output_path}', 'metrics')")
                elif args.format == 'csv':
                    log_message(f"   df = pd.read_csv('{output_path}')")
                elif args.format == 'feather':
                    log_message(f"   df = pd.read_feather('{output_path}')")
                
                return 0
            else:
                log_message("❌ Conversion failed")
                return 1
                
    except KeyboardInterrupt:
        log_message("\n⏹️  Conversion interrupted by user")
        return 1
    except Exception as e:
        log_message(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())