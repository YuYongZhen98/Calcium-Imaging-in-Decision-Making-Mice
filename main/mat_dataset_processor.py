#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAT数据集处理器 - 跨平台完整版本
功能：处理原始MAT文件，提取fMRI数据和行为标签，生成标准化数据格式
适用系统：Windows, Linux, macOS
用法：python3 mat_dataset_processor.py [选项]
"""

import os
import sys
import re
import argparse
import h5py
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MATDatasetProcessor:
    """
    MAT数据集处理器核心类
    
    主要功能：
    1. 加载原始MAT格式的fMRI数据
    2. 提取行为标签（Frequency, Action, Reward等）
    3. 将数据转换为统一的四维格式 (trials × Cells × frames × labels)
    4. 保存处理后的数据和标签到文件
    """
    
    def __init__(self, input_folder: Union[str, Path], output_folder: Union[str, Path] = None):
        """
        初始化处理器
        
        Args:
            input_folder: 输入文件夹路径，包含原始MAT文件
            output_folder: 输出文件夹路径，如果不指定则自动创建
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
            
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'start_time': datetime.now()
        }
        
        print("=" * 60)
        print("MAT数据集处理器 - 跨平台完整版本")
        print("=" * 60)
        print(f"输入文件夹: {self.input_folder.absolute()}")
        print(f"输出文件夹: {self.output_folder.absolute()}")
        print(f"开始时间: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    @staticmethod
    def natural_sort_key(s: str) -> List[Union[int, str]]:
        """自然排序函数，确保文件按数字顺序排列"""
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', str(s))]
    
    @staticmethod
    def extract_session_number(file_name: str) -> int:
        """从文件名中提取会话编号"""
        match = re.search(r'Sess(\d+)', file_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0
    
    def find_original_mat_files(self) -> List[Path]:
        """查找所有需要处理的原始MAT文件"""
        mat_files = []
        
        for file_path in self.input_folder.glob("*.mat"):
            file_name = file_path.name
            if 'Sess' in file_name and 'Passive' not in file_name:
                mat_files.append(file_path)
        
        mat_files.sort(key=lambda x: self.natural_sort_key(x.name))
        self.stats['total_files'] = len(mat_files)
        
        if mat_files:
            print(f"找到 {len(mat_files)} 个原始MAT文件:")
            for i, file_path in enumerate(mat_files, 1):
                file_size = file_path.stat().st_size / 1024  # KB
                print(f"  {i:2d}. {file_path.name} ({file_size:.1f} KB)")
        else:
            print(f"警告: 在 {self.input_folder} 中未找到符合条件的原始MAT文件")
        
        return mat_files
    
    def load_original_mat_file(self, file_path: Path) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """加载单个原始MAT文件"""
        file_name = file_path.name
        
        if file_name.startswith('Passive') or file_name.startswith('processed_'):
            print(f"  跳过已处理的文件: {file_name}")
            self.stats['skipped_files'] += 1
            return None
        
        print(f"正在处理: {file_name}")
        
        try:
            # 检查文件是否存在且可读
            if not file_path.exists():
                print(f"  错误: 文件不存在: {file_name}")
                self.stats['failed_files'] += 1
                return None
            
            if not os.access(file_path, os.R_OK):
                print(f"  错误: 文件不可读: {file_name}")
                self.stats['failed_files'] += 1
                return None
            
            with h5py.File(file_path, 'r') as f:
                if 'data_aligned' not in f or 'behavResults' not in f:
                    print(f"  错误: {file_name} 中缺少必需字段")
                    self.stats['failed_files'] += 1
                    return None
                
                data_aligned = f['data_aligned'][()]
                data_aligned_matlab = np.transpose(data_aligned, (2, 1, 0))
                
                behav_group = f['behavResults']
                labels: Dict[str, np.ndarray] = {}
                required_vars = ['Stim_toneFreq', 'Trial_Type', 'Action_choice', 'Time_reward']
                rename_required_vars = ['Stim_toneFreq', 'Frequency', 'Action', 'Time_reward']
                
                for rename_var_name,var_name in zip(rename_required_vars, required_vars):
                    if var_name in behav_group:
                        data = np.transpose(behav_group[var_name][()], (1, 0))
                        if data.ndim == 2:
                            if data.shape[0] == 1:
                                data = data.flatten()
                            elif data.shape[1] == 1:
                                data = data.flatten()
                        labels[rename_var_name] = data
                        
                    else:
                        print(f"  警告: {file_name} 中缺少标签 '{var_name}'，用0填充")
                        labels[rename_var_name] = np.zeros(data_aligned_matlab.shape[0])
                
                action_choice = labels['Action']
                time_reward = labels['Time_reward']
                outcome = np.zeros(len(action_choice), dtype=np.int32)
                
                for i in range(len(action_choice)):
                    if action_choice[i] == 2:
                        outcome[i] = 2
                    elif action_choice[i] != 2 and time_reward[i] != 0:
                        outcome[i] = 1
                    elif action_choice[i] != 2 and time_reward[i] == 0:
                        outcome[i] = 0
                
                labels['Reward'] = outcome
                
                n_trials_matlab = data_aligned_matlab.shape[0]
                label_length = len(labels['Stim_toneFreq'])
                actual_trials = min(label_length, n_trials_matlab)
                
                labels['start_frame'] = f['start_frame'][()]
                labels['frame_rate'] = f['frame_rate'][()]
                
                
                data_dict: Dict[str, Any] = {
                    'data_aligned': data_aligned_matlab,
                    'labels': labels,
                    'actual_trials': actual_trials,
                    'n_cells': data_aligned_matlab.shape[1],
                    'n_frames': data_aligned_matlab.shape[2]
                }
                
                metadata_dict: Dict[str, Any] = {
                    'file_name': file_name,
                    'session_number': self.extract_session_number(file_name),
                    'n_trials': actual_trials,
                    'n_cells': data_aligned_matlab.shape[1],
                    'n_frames': data_aligned_matlab.shape[2],
                    'original_shape': data_aligned.shape,
                    'transposed_shape': data_aligned_matlab.shape
                }
                
                print(f"  成功加载: {file_name}")
                print(f"    可用trials: {actual_trials}, Cells: {data_aligned_matlab.shape[1]}, "
                      f"frames: {data_aligned_matlab.shape[2]}")
                return data_dict, metadata_dict
                
        except OSError as e:
            print(f"  文件I/O错误 {file_name}: {str(e)}")
            self.stats['failed_files'] += 1
            return None
        except Exception as e:
            print(f"  加载文件 {file_name} 时出错: {str(e)}")
            self.stats['failed_files'] += 1
            return None
    
    def create_Frequency_Action_Reward(self, data_dict: Dict[str, Any]) -> np.ndarray:
        """创建数据数组"""
        data_aligned = data_dict['data_aligned']
        labels = data_dict['labels']
        actual_trials = data_dict['actual_trials']
        
        n_cells = data_aligned.shape[1]
        n_frames = data_aligned.shape[2]
        Frequency_Action_Reward = np.zeros((actual_trials,  3), dtype=np.float32)
        
        trial_type = labels['Frequency']
        action_choice = labels['Action']
        outcome = labels['Reward']
        
        for i in range(actual_trials):
            Frequency_Action_Reward[i,  0] = trial_type[i] if i < len(trial_type) else 0
            Frequency_Action_Reward[i,  1] = action_choice[i] if i < len(action_choice) else 0
            Frequency_Action_Reward[i,  2] = outcome[i] if i < len(outcome) else 0
        
        return Frequency_Action_Reward
    
    def create_label_dataframe(self, data_dict: Dict[str, Any], metadata: Dict[str, Any]) -> pd.DataFrame:
        """创建标签DataFrame"""
        labels = data_dict['labels']
        actual_trials = data_dict['actual_trials']
        
        trial_ids = np.arange(1, actual_trials + 1)
        
        df_data = {
            'Trial_ID': trial_ids,
            'File_Name': metadata['file_name'],
            'Session_ID': metadata['session_number']
        }
        
        label_keys = ['Stim_toneFreq', 'Frequency', 'Action', 'Reward']
        for label_name in label_keys:
            if label_name in labels:
                label_values = labels[label_name]
                if len(label_values) >= actual_trials:
                    df_data[label_name] = label_values[:actual_trials]
                else:
                    padded = np.zeros(actual_trials)
                    padded[:len(label_values)] = label_values
                    df_data[label_name] = padded
        return pd.DataFrame(df_data)
    
    def process_single_file(self, file_path: Path, save_label: bool = True) -> Optional[Dict[str, Any]]:
        """处理单个MAT文件"""
        file_name = file_path.name
        
        if file_name.startswith('processed_'):
            print(f"  跳过已处理的文件: {file_name}")
            self.stats['skipped_files'] += 1
            return None
        
        loaded_data = self.load_original_mat_file(file_path)
        if loaded_data is None:
            return None
        
        data_dict, metadata = loaded_data
        df = self.create_label_dataframe(data_dict, metadata)
        Frequency_Action_Reward = self.create_Frequency_Action_Reward(data_dict)
        
        result_dict = None
        if save_label:
            processed_file_name = f"processed_{file_name}"
            processed_mat_path = self.output_folder / processed_file_name
            
            save_dict = {
                'Frequency_Action_Reward': data_dict['data_aligned'][:data_dict['actual_trials'], :, :],
                'Labels': Frequency_Action_Reward,
                'Original_file': file_name,
                'Start_frame':data_dict['labels']['start_frame'][:data_dict['actual_trials']],
                'Frame_rate':data_dict['labels']['frame_rate'][:data_dict['actual_trials']],
                'Stim_toneFreq':data_dict['labels']['Stim_toneFreq'][:data_dict['actual_trials']]
            }
            
            # for key in ['Stim_toneFreq', 'Frequency', 'Action', 'Reward']:
            #     if key in data_dict['labels']:
            #         values = data_dict['labels'][key]
            #         if len(values) > data_dict['actual_trials']:
            #             save_dict[key] = values[:data_dict['actual_trials']]
            #         else:
            #             save_dict[key] = values
            
            try:
                sio.savemat(processed_mat_path, save_dict, do_compression=True, long_field_names=True)
                print(f"    数据已保存: {processed_file_name}")
            except Exception as e:
                print(f"    保存数据失败: {str(e)}")
                save_label = False
        
        self.stats['successful_files'] += 1
        
        result_dict = {
            'file_name': file_name,
            'session_number': metadata['session_number'],
            'dataframe': df,
            'Frequency_Action_Reward': Frequency_Action_Reward if save_label else None,
            'metadata': metadata,
            'data_dict': data_dict,
            'label_saved': save_label
        }
        
        return result_dict
    
    def process_all_files_to_excel(self, output_excel_path: str = None) -> pd.DataFrame:
        """批量处理所有MAT文件并保存标签到Excel"""
        print("\n" + "=" * 60)
        print("步骤1: 提取所有原始MAT文件的标签信息")
        print("=" * 60)
        
        mat_files = self.find_original_mat_files()
        
        if not mat_files:
            print("没有找到可处理的原始MAT文件")
            return pd.DataFrame()
        
        if output_excel_path and os.path.exists(output_excel_path):
            print(f"注意: 输出文件 {output_excel_path} 已存在，将被覆盖")
        
        all_dataframes = []
        all_metadata = []
        
        if output_excel_path:
            output_dir = Path(output_excel_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                with pd.ExcelWriter(output_excel_path, engine='openpyxl', mode='w') as excel_writer:
                    print(f"输出Excel文件: {output_excel_path}")
                    
                    for i, mat_file in enumerate(tqdm(mat_files, desc="处理MAT文件", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')):
                        file_name = mat_file.name
                        print(f"\n[{i+1}/{len(mat_files)}] 处理文件: {file_name}")
                        
                        result = self.process_single_file(mat_file, save_label=False)
                        
                        if result is not None:
                            df = result['dataframe']
                            metadata = result['metadata']
                            
                            all_dataframes.append(df)
                            all_metadata.append(metadata)
                            
                            session_num = metadata['session_number']
                            sheet_name = f"Sess{session_num}"
                            if len(sheet_name) > 31:
                                sheet_name = sheet_name[:31]
                            df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
                            print(f"    已添加到Excel工作表: {sheet_name}")
                        else:
                            print(f"    处理失败: {file_name}")
                    
                    if all_dataframes:
                        print("\n创建汇总工作表...")
                        
                        combined_df = pd.concat(all_dataframes, ignore_index=True)
                        
                        summary_data = []
                        for metadata in all_metadata:
                            summary_data.append({
                                'Session_ID': metadata.get('session_number', 0),
                                'File_Name': metadata.get('file_name', 'Unknown'),
                                'Trials': metadata.get('n_trials', 0),
                                'Cells': metadata.get('n_cells', 0),
                                'Frames': metadata.get('n_frames', 0)
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(excel_writer, sheet_name='File_Summary', index=False)
                        
                        label_stats = []
                        for label_col in ['Stim_toneFreq', 'Frequency', 'Action', 'Reward']:
                            if label_col in combined_df.columns:
                                unique_vals = combined_df[label_col].unique()
                                label_stats.append({
                                    'Label': label_col,
                                    'Unique_Values': len(unique_vals),
                                    'Min': combined_df[label_col].min() if pd.api.types.is_numeric_dtype(combined_df[label_col]) else 'N/A',
                                    'Max': combined_df[label_col].max() if pd.api.types.is_numeric_dtype(combined_df[label_col]) else 'N/A',
                                    'Mean': combined_df[label_col].mean() if pd.api.types.is_numeric_dtype(combined_df[label_col]) else 'N/A'
                                })
                        
                        stats_df = pd.DataFrame(label_stats)
                        stats_df.to_excel(excel_writer, sheet_name='Label_Statistics', index=False)
                        
                        print(f"Excel文件已保存: {output_excel_path}")
            
            except PermissionError:
                print(f"错误: 无法写入文件，可能文件已被打开: {output_excel_path}")
                return pd.DataFrame()
            except Exception as e:
                print(f"保存Excel文件时出错: {str(e)}")
                return pd.DataFrame()
        else:
            # 如果不保存到Excel，仍然处理文件收集数据
            for i, mat_file in enumerate(tqdm(mat_files, desc="处理MAT文件", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')):
                file_name = mat_file.name
                print(f"\n[{i+1}/{len(mat_files)}] 处理文件: {file_name}")
                
                result = self.process_single_file(mat_file, save_label=False)
                
                if result is not None:
                    df = result['dataframe']
                    metadata = result['metadata']
                    
                    all_dataframes.append(df)
                    all_metadata.append(metadata)
                else:
                    print(f"    处理失败: {file_name}")
        
        print("\n" + "=" * 60)
        print("Excel处理完成!")
        print("=" * 60)
        print(f"总文件数: {self.stats['total_files']}")
        print(f"成功处理: {self.stats['successful_files']}")
        print(f"处理失败: {self.stats['failed_files']}")
        print(f"跳过文件: {self.stats['skipped_files']}")
        
        if all_dataframes:
            
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"总Trials数: {len(combined_df)}")
            print(f"总Sessions数: {combined_df['Session_ID'].nunique()}")
            
            if 'Reward' in combined_df.columns:
                outcome_counts = combined_df['Reward'].value_counts().sort_index()
                print("Reward:")
                for outcome, count in outcome_counts.items():
                    percentage = count / len(combined_df) * 100
                    print(f"  Reward {outcome}: {count} trials ({percentage:.1f}%)")
            
            return combined_df
        return pd.DataFrame()
    
    def generate_label_files(self):
        """为所有原始MAT文件生成数据文件"""
        print("\n" + "=" * 60)
        print("步骤2: 为所有原始MAT文件生成数据")
        print("=" * 60)
        
        mat_files = self.find_original_mat_files()
        
        if not mat_files:
            print("没有找到可处理的原始MAT文件")
            return
        
        print(f"准备为 {len(mat_files)} 个文件生成数据...")
        
        processed_count = 0
        
        for mat_file in tqdm(mat_files, desc="生成四维MAT文件", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            file_name = mat_file.name
            
            if file_name.startswith('Passive') or file_name.startswith('processed_'):
                print(f"  跳过已处理的文件: {file_name}")
                continue
            
            print(f"\n处理文件: {file_name}")
            
            result = self.process_single_file(mat_file, save_label=True)
            
            if result is not None and result['label_saved']:
                processed_count += 1
        
        print("\n数据生成完成!")
        print(f"输出文件夹: {self.output_folder.absolute()}")
        print(f"成功生成: {processed_count} 个文件")
    
    def clean_processed_files(self) -> int:
        """清理已存在的processed_*.mat文件"""
        print("清理已存在的processed_*.mat文件...")
        
        deleted_count = 0
        for file_path in self.output_folder.glob("processed_*.mat"):
            try:
                file_size = file_path.stat().st_size / 1024
                file_path.unlink()
                print(f"  已删除: {file_path.name} ({file_size:.1f} KB)")
                deleted_count += 1
            except Exception as e:
                print(f"  删除失败 {file_path.name}: {e}")
        
        print(f"共删除 {deleted_count} 个已处理文件")
        return deleted_count
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats.copy()
    
    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "=" * 60)
        print("处理摘要")
        print("=" * 60)
        print(f"输入文件夹: {self.input_folder.absolute()}")
        print(f"输出文件夹: {self.output_folder.absolute()}")
        print(f"总文件数: {self.stats['total_files']}")
        print(f"成功处理: {self.stats['successful_files']}")
        print(f"处理失败: {self.stats['failed_files']}")
        print(f"跳过文件: {self.stats['skipped_files']}")
        
        if 'start_time' in self.stats:
            end_time = datetime.now()
            duration = end_time - self.stats['start_time']
            print(f"开始时间: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"处理时长: {duration}")
        
        print("=" * 60)


def process_dataset(input_folder: str, output_excel: str = None, 
                    no_excel: bool = False, no_label: bool = False,
                    clean_only: bool = False, output_folder: str = None):
    """处理单个数据集的函数"""
    print(f"\n{'='*80}")
    print(f"开始处理数据集: {input_folder}")
    print(f"{'='*80}")
    
    # 创建处理器实例
    processor = MATDatasetProcessor(
        input_folder=input_folder,
        output_folder=output_folder
    )
    
    # 只执行清理操作
    if clean_only:
        print("执行清理操作...")
        deleted_count = processor.clean_processed_files()
        print(f"清理完成，共删除 {deleted_count} 个文件")
        return
    
    # 处理流程
    try:
        # 1. 清理已存在的processed文件
        print("\n1. 清理已存在的processed文件...")
        processor.clean_processed_files()
        
        # 2. 提取标签并保存到Excel（除非指定不生成）
        if not no_excel:
            print("\n2. 提取标签并保存到Excel...")
            combined_df = processor.process_all_files_to_excel(
                output_excel_path=output_excel
            )
        
        # 3. 生成数据文件（除非指定不生成）
        if not no_label:
            print("\n3. 生成数据文件...")
            processor.generate_label_files()
        
        # 4. 打印处理摘要
        processor.print_summary()
        
        print(f"\n{'='*80}")
        print(f"数据集处理完成: {input_folder}")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
        raise
    except Exception as e:
        print(f"\n处理数据集 {input_folder} 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def check_dependencies() -> bool:
    """
    检查所有必要的依赖包是否已安装
    
    返回:
        bool: 如果所有依赖都满足返回True，否则返回False
    """
    # 必需的核心依赖包
    required_packages = {
        'pandas': ('pandas', '1.5.0'),
        'numpy': ('numpy', '1.23.0'),
        'h5py': ('h5py', '3.7.0'),
        'scipy': ('scipy', '1.9.0'),
        'tqdm': ('tqdm', '4.64.0'),
        'openpyxl': ('openpyxl', '3.0.0')  # pandas写入Excel需要
    }
    
    # 可选的图表依赖包（如果缺失，只生成警告）
    optional_packages = {
        'matplotlib': ('matplotlib', '3.5.0')  # 图表生成需要，但如果不生成图表则非必需
    }
    
    missing_required = []
    missing_optional = []
    
    print("检查依赖包...")
    
    # 检查必需依赖包
    for display_name, (import_name, min_version) in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_required.append(f"{display_name}>={min_version}")
            print(f"  {display_name:12s} 未安装          (需要 >= {min_version})")
        else:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', '未知')
                print(f"  {display_name:12s} {version:12s} (需要 >= {min_version})")
            except AttributeError:
                print(f"   {display_name:12s} 版本未知        (需要 >= {min_version})")
    
    # 检查可选依赖包
    for display_name, (import_name, min_version) in optional_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_optional.append(f"{display_name}>={min_version}")
            print(f"  {display_name:12s} 未安装          (可选，用于图表生成)")
        else:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', '未知')
                print(f"  {display_name:12s} {version:12s} (可选，需要 >= {min_version})")
            except AttributeError:
                print(f"  {display_name:12s} 版本未知        (可选，需要 >= {min_version})")
    
    # 如果有缺失的必需包，显示错误信息
    if missing_required:
        print("\n" + "=" * 60)
        print("错误: 缺少必要的Python库")
        print("=" * 60)
        print(f"缺失的必需包: {', '.join(missing_required)}")
        print("\n请运行以下命令安装所有依赖:")
        print("pip install pandas numpy h5py scipy tqdm openpyxl matplotlib")
        print("\n或创建一个requirements.txt文件并运行:")
        print("pip install -r requirements.txt")
        print("=" * 60)
        return False
    
    # 如果有缺失的可选包，显示警告信息
    if missing_optional:
        print("\n" + "=" * 60)
        print("警告: 缺少可选的Python库")
        print("=" * 60)
        print(f"缺失的可选包: {', '.join(missing_optional)}")
        print("\n注意: 缺少这些包将无法生成统计图表")
        print("如果需要生成图表，请运行以下命令安装:")
        print("pip install matplotlib")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("依赖包检查完成!")
    print("必需依赖包: 全部满足")
    print("可选依赖包: " + ("全部满足" if not missing_optional else "部分缺失"))
    print("=" * 60)
    return True


def read_excel_files_and_calculate_stats(excel_files):
    """
    读取Excel文件并计算统计数据
    
    返回包含数据集统计信息的字典
    """
    datasets = {}
    
    for excel_path in excel_files:
        dataset_name = Path(excel_path).parent.name
        
        print(f"\n处理数据集: {dataset_name}")
        print(f"Excel文件路径: {excel_path}")
        
        if not os.path.exists(excel_path):
            print(f"警告: 文件不存在 - {excel_path}")
            continue
        
        try:
            # 读取Excel文件
            excel_data = pd.ExcelFile(excel_path)
            
            # 获取所有工作表名称
            sheet_names = excel_data.sheet_names
            
            # 过滤出session相关的工作表
            session_sheets = [sheet for sheet in sheet_names if sheet.startswith('Sess')]
            
            all_dataframes = []
            for sheet in session_sheets:
                try:
                    df = pd.read_excel(excel_path, sheet_name=sheet)
                    all_dataframes.append(df)
                except Exception as e:
                    print(f"读取工作表 {sheet} 失败: {e}")
            
            if all_dataframes:
                combined_df = pd.concat(all_dataframes, ignore_index=True)
                
                # 读取File_Summary工作表
                summary_df = pd.read_excel(excel_path, sheet_name='File_Summary')
                
                # 计算统计信息
                if 'Session_ID' in combined_df.columns:
                    unique_sessions = combined_df['Session_ID'].nunique()
                else:
                    unique_sessions = len(session_sheets)
                
                total_trials = len(combined_df)
                avg_trials = total_trials / unique_sessions if unique_sessions > 0 else 0
                
                # 计算目标列的统计信息
                target_columns = ['Frequency', 'Action', 'Reward']
                columns_stats = {}
                
                for col in target_columns:
                    if col in combined_df.columns:
                        col_data = pd.to_numeric(combined_df[col], errors='coerce').dropna()
                        value_counts = col_data.value_counts().sort_index()
                        
                        value_counts_avg = {}
                        for value, count in value_counts.items():
                            avg_count = count / unique_sessions if unique_sessions > 0 else 0
                            value_counts_avg[value] = avg_count
                        
                        columns_stats[col] = {
                            'value_counts': value_counts.to_dict(),
                            'value_counts_avg': value_counts_avg,
                        }
                
                # 计算Cells和Frames的统计
                cells_total = pd.to_numeric(summary_df['Cells'], errors='coerce').sum()
                frames_total = pd.to_numeric(summary_df['Frames'], errors='coerce').sum()
                
                cells_total = int(cells_total)
                frames_total = int(frames_total)
                
                avg_cells = float(cells_total / unique_sessions) if unique_sessions > 0 else 0.0
                avg_frames = float(frames_total / unique_sessions) if unique_sessions > 0 else 0.0
                
                datasets[dataset_name] = {
                    'sessions': int(unique_sessions),
                    'avg_trials': float(avg_trials),
                    'total_trials': int(total_trials),
                    'columns_stats': columns_stats,
                    'cells_total': cells_total,
                    'avg_cells': avg_cells,
                    'frames_total': frames_total,
                    'avg_frames': avg_frames,
                }
                
                print(f"  Session数量: {unique_sessions}")
                print(f"  总Trial数量: {total_trials}")
                print(f"  平均Trial数/Session: {avg_trials:.1f}")
        
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            datasets[dataset_name] = None
    
    return datasets


def plot_all_metrics_grouped(datasets):
    """
    分组绘制所有指标，使图表更清晰
    
    保存为PNG和PDF格式
    """
    if not datasets:
        print("没有可用的数据集数据")
        return
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建大图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 提取数据集名称
    dataset_names = list(datasets.keys())
    
    # 提取数据集名称
    dataset_names = list(datasets.keys())
    
    # 定义所有要绘制的指标（按组排列）
    metric_groups = [
        'Basic Statistics',
        'Frequency',
        'Action', 
        'Reward'
    ]
    
    # 将指标分组
    metric_groups = {
        'Basic Statistics': ['sessions', 'avg_trials', 'avg_cells', 'avg_frames'],
        'Frequency': ['Frequency_0', 'Frequency_1'],
        'Action': ['Action_0', 'Action_1', 'Action_2'],
        'Reward': ['Reward_0', 'Reward_1', 'Reward_2']
    }
    
    # 准备数据
    all_metrics_data = {}
    for metric in [item for sublist in metric_groups.values() for item in sublist]:
        values = []
        for dataset in dataset_names:
            if metric in ['sessions', 'avg_trials', 'avg_cells', 'avg_frames']:
                values.append(datasets[dataset][metric])
            else:
                metric_type = metric.rsplit('_', 1)[0]
                metric_value = int(metric.split('_')[-1])
                values.append(datasets[dataset]['columns_stats'][metric_type]['value_counts_avg'][metric_value])
        all_metrics_data[metric] = values
    
    # 设置柱状图的位置
    x = np.arange(len(dataset_names))  # 数据集位置
    total_metrics = sum(len(metrics) for metrics in metric_groups.values())
    group_width = 0.8  # 每组的总宽度
    bar_width = group_width / total_metrics  # 每个柱子的宽度
    
    group_colors = {
        'Basic Statistics': ['#1f77b4', '#d62728', '#ff7f0e', '#9467bd'],
        'Frequency': ['#8c564b', '#c49c94'],
        'Action': ['#1b9e77', '#66c2a4', '#b2e2e2'],
        'Reward': ['#e7298a', '#c994c7', '#df65b0']
    }
    
    # 计算起始位置
    start_pos = x - group_width/2
    
    # 绘制分组柱状图
    bar_idx = 0
    for group_name, metrics in metric_groups.items():
        group_color_scheme = group_colors[group_name]
        
        for j, metric in enumerate(metrics):
            values = all_metrics_data[metric]
            positions = start_pos + bar_idx * bar_width + bar_width/2
            
            color_idx = j % len(group_color_scheme)
            color = group_color_scheme[color_idx]
            
            bars = ax.bar(positions, values, bar_width * 0.9,
                         color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            # 在柱子上添加数值标签
            for k, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                offset = max(values) * 0.02 if max(values) > 0 else 1
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{value:.1f}' if value < 10 else f'{int(value)}',
                       ha='center', va='bottom', fontsize=8)
            
            bar_idx += 1
    
    # 设置图表属性
    ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
    ax.set_ylabel('Numerical value', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Dataset Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=12)
    
    # 添加网格线
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # 设置Y轴范围
    max_value = max([max(v) for v in all_metrics_data.values()])
    ax.set_ylim(0, max_value * 1.15)
    
    # 创建自定义图例
    from matplotlib.patches import Patch
    legend_elements = []
    
    legend_mapping = {
        'sessions': 'Sessions',
        'avg_trials': 'Trials',
        'avg_cells': 'Cells',
        'avg_frames': 'Frames',
        'Frequency_0': 'Low',
        'Frequency_1': 'High',
        'Action_0': 'Left',
        'Action_1': 'Right',
        'Action_2': 'No_Action',
        'Reward_0': 'Error',
        'Reward_1': 'Correct',
        'Reward_2': 'Miss'
    }
    
    for group_name, metrics in metric_groups.items():
        group_color_scheme = group_colors[group_name]
        for j, metric in enumerate(metrics):
            display_name = legend_mapping.get(metric, metric)
            color_idx = j % len(group_color_scheme)
            color = group_color_scheme[color_idx]
            legend_elements.append(Patch(facecolor=color, label=display_name, alpha=0.8))
    
    # 添加图例
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1),
              fontsize=10, title='Metrics', title_fontsize=11, ncol=4)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 保存图片
    plt.savefig('comparison_of_dataset_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('comparison_of_dataset_metrics.pdf', bbox_inches='tight')
    
    plt.show()


def plot_transposed_metrics(datasets):
    """
    绘制转置的分组图表：X轴为指标，图例为数据集
    
    保存为PNG和PDF格式
    """
    if not datasets:
        print("没有可用的数据集数据")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建大图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 提取数据集名称
    dataset_names = list(datasets.keys())
    
    # 定义所有要绘制的指标（按组排列）
    metric_groups = [
        'Basic Statistics',
        'Frequency',
        'Action', 
        'Reward'
    ]
    
    # 每个组内的具体指标
    metrics_by_group = {
        'Basic Statistics': ['sessions', 'avg_trials', 'avg_cells', 'avg_frames'],
        'Frequency': ['Frequency_0', 'Frequency_1'],
        'Action': ['Action_0', 'Action_1', 'Action_2'],
        'Reward': ['Reward_0', 'Reward_1', 'Reward_2']
    }
    
    # 指标标签（用于显示）
    metric_labels = {
        'sessions': 'Sessions',
        'avg_trials': 'Trials',
        'avg_cells': 'Cells',
        'avg_frames': 'Frames',
        'Frequency_0': 'Low',
        'Frequency_1': 'High',
        'Action_0': 'Left',
        'Action_1': 'Right',
        'Action_2': 'No_Action',
        'Reward_0': 'Error',
        'Reward_1': 'Correct',
        'Reward_2': 'Miss'
    }
    
    # 构建所有指标列表（按组顺序）
    all_metrics = []
    for group in metric_groups:
        all_metrics.extend(metrics_by_group[group])
    
    # 颜色方案 - 为每个数据集分配颜色
    dataset_colors = {
        '7_28kHz_Data': '#FF6B6B',
        '4_16kHz_Data': '#4ECDC4',
        '8_32kHz_Data': '#45B7D1'
    }
    
    # 辅助函数：获取指标值
    def get_metric_value(dataset_name, metric):
        dataset = datasets[dataset_name]
        
        if metric in ['sessions', 'avg_trials', 'avg_cells', 'avg_frames']:
            return dataset[metric]
        
        if metric.startswith('Frequency_'):
            value = int(metric.split('_')[-1])
            return dataset['columns_stats']['Frequency']['value_counts_avg'][value]
        elif metric.startswith('Action_'):
            value = int(metric.split('_')[-1])
            return dataset['columns_stats']['Action']['value_counts_avg'][value]
        elif metric.startswith('Reward_'):
            value = int(metric.split('_')[-1])
            return dataset['columns_stats']['Reward']['value_counts_avg'][value]
        
        return 0
    
    # 设置柱状图的位置
    x = np.arange(len(all_metrics))
    width = 0.25
    total_width = len(dataset_names) * width
    
    # 绘制每个数据集的柱子
    for i, dataset_name in enumerate(dataset_names):
        values = [get_metric_value(dataset_name, metric) for metric in all_metrics]
        positions = x + i * width - total_width/2 + width/2
        
        bars = ax.bar(positions, values, width, 
                     label=dataset_name, 
                     color=dataset_colors.get(dataset_name, '#999999'),
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1)
        
        # 在柱子上添加数值标签
        for k, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            offset = max(values) * 0.02 if max(values) > 0 else 1
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{value:.1f}' if value < 10 else f'{int(value)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 设置X轴标签
    x_labels = [metric_labels.get(metric, metric) for metric in all_metrics]
    ax.set_xlabel('Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('Numerical value', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, ha='center', fontsize=10)
    
    # 添加分组标签
    group_boundaries = [0]
    current_index = 0
    for group in metric_groups:
        current_index += len(metrics_by_group[group])
        group_boundaries.append(current_index - 0.5)
    
    # 添加垂直线分隔不同组
    for boundary in group_boundaries[1:-1]:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # 添加组标签
    current_index = 0
    metric_regroups = [
        'Basic Statistics',
        'Frequency',
        'Action', 
        'Reward'
    ]
    for i, group in enumerate(metric_groups):
        group_size = len(metrics_by_group[group])
        group_center = current_index + group_size / 2 - 0.5
        ax.text(group_center, ax.get_ylim()[1] * 1.02, metric_regroups[i], 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        current_index += group_size
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加网格线
    ax.grid(True, alpha=0.3, axis='y')
    
    # 调整Y轴范围
    y_max = max([get_metric_value(name, metric) 
                 for name in dataset_names 
                 for metric in all_metrics])
    ax.set_ylim(0, y_max * 1.15)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('transposed_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('transposed_metrics_comparison.pdf', bbox_inches='tight')
    plt.show()


def main() -> None:
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(
        description='MAT数据集处理器 - 处理原始MAT文件，提取fMRI数据和行为标签',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认的三个数据集路径处理所有数据
  python3 mat_dataset_processor.py
  
  # 只处理指定文件夹
  python3 mat_dataset_processor.py -i ./Data/21Sessions_Data
  
  # 指定输出Excel文件
  python3 mat_dataset_processor.py -i ./Data/21Sessions_Data -o ./results/labels.xlsx
  
  # 只生成数据，不生成Excel
  python3 mat_dataset_processor.py -i ./Data/21Sessions_Data --no-excel
  
  # 只生成Excel，不生成数据
  python3 mat_dataset_processor.py -i ./Data/21Sessions_Data --no_label
  
  # 清理已处理的文件
  python3 mat_dataset_processor.py -i ./Data/21Sessions_Data --clean-only
  
  # 显示详细帮助
  python3 mat_dataset_processor.py -h
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='输入文件夹路径，包含原始MAT文件（如果不指定，则使用默认的三个数据集）'
    )
    
    parser.add_argument(
        '-o', '--output-excel',
        type=str,
        default=None,
        help='输出Excel文件路径（仅当使用单个数据集时有效）'
    )
    
    parser.add_argument(
        '--no-excel',
        action='store_true',
        help='不生成Excel文件'
    )
    
    parser.add_argument(
        '--no_label',
        action='store_true',
        help='不生成数据文件'
    )
    
    parser.add_argument(
        '--clean-only',
        action='store_true',
        help='只清理已处理的文件，不进行其他处理'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help='数据输出文件夹（默认：输入文件夹/processed_mat_files）'
    )
    
    args = parser.parse_args()
    
    # 默认数据集路径（使用正斜杠确保跨平台兼容）
    DEFAULT_INPUT_FOLDERS = [
        "../Data/Dataset for Figures 2, 4, 6, 7, 8/21Sessions_Data",
        "../Data/Dataset for Figures 5 and 8F/4_16Sess", 
        "../Data/Dataset for Figures 5 and 8F/7_28Sess"
    ]
    
            
    DEFAULT_OUTPUT_FOLDERS = [
        "./8_32kHz_Data/processed_mat_files",
        "./4_16kHz_Data/processed_mat_files",
        "./7_28kHz_Data/processed_mat_files"
    ]
    
    DEFAULT_OUTPUT_EXCELS = [
        "./8_32kHz_Data/all_sessions_labels.xlsx",
        "./4_16kHz_Data/all_sessions_labels.xlsx",
        "./7_28kHz_Data/all_sessions_labels.xlsx"
    ]
    
    # 检查输入路径是否存在
    def check_path_exists(path_str: str) -> bool:
        path = Path(path_str)
        if not path.exists():
            print(f"警告: 路径不存在: {path.absolute()}")
            return False
        return True
    
    # 如果没有指定输入文件夹，则使用默认的三个数据集
    if args.input is None:
        print("=" * 80)
        print("未指定输入文件夹，使用默认的三个数据集路径")
        print("=" * 80)

        
        # 检查所有默认路径是否存在
        valid_datasets = []
        for i, input_folder in enumerate(DEFAULT_INPUT_FOLDERS):
            if check_path_exists(input_folder):
                valid_datasets.append({
                    'input': input_folder,
                    'output_excel': DEFAULT_OUTPUT_EXCELS[i] if i < len(DEFAULT_OUTPUT_EXCELS) else None,
                    'output_folder': DEFAULT_OUTPUT_FOLDERS[i] 
                })
        
        if not valid_datasets:
            print("错误: 默认数据集路径都不存在！")
            print("请检查以下路径:")
            for i, folder in enumerate(DEFAULT_INPUT_FOLDERS):
                print(f"  {i+1}. {folder}")
            sys.exit(1)
        
        print(f"将处理 {len(valid_datasets)} 个数据集")
        
        # 处理所有数据集
        for i, dataset in enumerate(valid_datasets):
            print(f"\n处理第 {i+1}/{len(valid_datasets)} 个数据集")
            print(f"输入: {dataset['input']}")
            if dataset['output_excel'] and not args.no_excel:
                print(f"输出Excel: {dataset['output_excel']}")
            
            try:
                process_dataset(
                    input_folder=dataset['input'],
                    output_excel=dataset['output_excel'],
                    no_excel=args.no_excel,
                    no_label=args.no_label,
                    clean_only=args.clean_only,
                    output_folder=dataset['output_folder']
                )
            except Exception as e:
                print(f"处理数据集时出错: {e}")
                continue
        
        print("\n" + "=" * 80)
        print("所有数据集处理完成!")
        print("=" * 80)
        
        # 生成PDF图表

            
        print("\n正在生成数据集统计图表...")
        
        excel_paths = [dataset['output_excel'] for dataset in valid_datasets if dataset['output_excel']]
        
        if excel_paths:
            datasets_stats = read_excel_files_and_calculate_stats(excel_paths)
            if datasets_stats:
                print("\n正在绘制分组图表...")
                plot_all_metrics_grouped(datasets_stats)
                
                print("\n正在绘制转置图表...")
                plot_transposed_metrics(datasets_stats)
                
                print("图表已保存为:")
                print("  - comparison_of_dataset_metrics.png")
                print("  - comparison_of_dataset_metrics.pdf")
                print("  - transposed_metrics_comparison.png")
                print("  - transposed_metrics_comparison.pdf")
        

        
    else:
        # 处理单个指定的数据集
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 输入文件夹不存在: {input_path.absolute()}")
            sys.exit(1)
        
        if not input_path.is_dir():
            print(f"错误: 输入路径不是文件夹: {input_path.absolute()}")
            sys.exit(1)
        
        # 设置默认输出Excel路径（如果不指定且需要生成Excel）
        if args.output_excel is None and not args.no_excel:
            folder_name = input_path.name
            args.output_excel = f"./{folder_name}/all_sessions_labels.xlsx"
            print(f"注意: 未指定输出Excel路径，使用默认: {args.output_excel}")
        
        process_dataset(
            input_folder=args.input,
            output_excel=args.output_excel,
            no_excel=args.no_excel,
            no_label=args.no_label,
            clean_only=args.clean_only,
            output_folder=args.output_folder
        )


if __name__ == "__main__":
    # 检查所有必要的依赖包
    if not check_dependencies():
        sys.exit(1)
    
    # 运行主程序
    main()