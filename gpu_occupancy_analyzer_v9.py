#!/usr/bin/env python3
"""
gem5 GPU Occupancy Analyzer
Calculates GPU occupancy metrics similar to Nsight Compute from gem5 stats.txt
"""

import re
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path
import math

class GPUOccupancyAnalyzer:
    def __init__(self, stats_file_path):
        self.stats_file = stats_file_path
        self.stats = {}
        self.occupancy_data = {}

        # GPU architecture parameters
        self.gpu_config = {
            'max_warps_per_sm': 40,      # Maximum warps per SM (typical for modern GPUs)
            'warp_size': 64,             # AMD wavefront size (64) vs NVIDIA warp size (32)
            'max_threads_per_sm': 2560,  # Maximum threads per SM
            'num_sms': 4,               # Number of streaming multiprocessors/compute units #64
            'max_blocks_per_sm': 40,     # Maximum blocks per SM. changed from 32
            'shared_memory_per_sm': 65536, # Shared memory per SM in bytes
            'registers_per_sm': 8192,   # Number of registers per SM
            'peak_mem_bandwidth_gbps': 50,  #default used to be 900 GB/s, but that is for modern gpus, since this is using gfx902 simulated apu, using a lower gbps is reasonable
        }

    def parse_stats_file(self):
        """Parse gem5 stats.txt file"""
        try:
            with open(self.stats_file, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: Stats file not found at {self.stats_file}")
            return False

        # Parse stats using regex patterns... changed original regex to include the chars after the :: as well
        #pattern = r'^([a-zA-Z0-9_.]+)\s+([0-9.e+-]+)(?:\s+#\s*(.*))?$'
        pattern = r'^([a-zA-Z0-9_.:]+)\s+([0-9.e+-]+)(?:\s+#\s*(.*))?$'


        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('---') or line.startswith('Begin'):
                continue

            match = re.match(pattern, line)
            if match:
                stat_name = match.group(1)
                stat_value = match.group(2)
                description = match.group(3) if match.group(3) else ""

                try:
                    if '.' in stat_value or 'e' in stat_value.lower():
                        stat_value = float(stat_value)
                    else:
                        stat_value = int(stat_value)
                except ValueError:
                    pass

                prev = self.stats.get(stat_name)
                if prev is None or stat_value > prev['value']:
                    self.stats[stat_name] = {'value': stat_value, 'description': description}
                #self.stats[stat_name] = {
                #    'value': stat_value,
                #    'description': description
                #}

        print(f"Parsed {len(self.stats)} statistics from gem5")
        return True

    def calculate_theoretical_occupancy(self):
        """Calculate theoretical occupancy based on kernel launch geometry and SM limits."""
        occupancy_metrics = {}

        # Use completed wavefronts and workgroups to infer kernel geometry
        total_wfs = self._get_total_wavefronts()
        total_wgs = self._get_total_workgroups()
        
        if not total_wfs or not total_wgs:
            return occupancy_metrics

        # Average wavefronts (warps) per workgroup (block)
        wavefronts_per_block = total_wfs / total_wgs

        warp_size = self.gpu_config['warp_size']              # 64 for gfx9
        max_threads_per_sm = self.gpu_config['max_threads_per_sm']
        max_blocks_per_sm = self.gpu_config['max_blocks_per_sm']

        # Threads per block implied by WFs per block
        threads_per_block = wavefronts_per_block * warp_size
        if threads_per_block <= 0:
            return occupancy_metrics
        active_threads_per_wave = (self._get_active_thread_percent() / 100) * self.gpu_config['warp_size']

        # Thread-based limit on active blocks per SM
        blocks_per_sm_threads = max_threads_per_sm // int(threads_per_block)
        if blocks_per_sm_threads <= 0:
            return occupancy_metrics

        # Actual active blocks per SM is limited by both threads and HW max blocks
        blocks_per_sm = min(blocks_per_sm_threads, max_blocks_per_sm)

        # --- Theoretical thread occupancy (% of max threads per SM) --- thread-based occupancy = (active warps per SM / maximum warps per SM)
        #print("total wfs: "+str(total_wfs)+"\ntotal wgs: "+str(total_wgs)+"\nwrap size: "+str(warp_size)+"\nblocks per sm: "+str(blocks_per_sm)+"\nmax threads per sm: "+str(max_threads_per_sm))
        thread_occ = ((active_threads_per_wave * wavefronts_per_block) * blocks_per_sm) / max_threads_per_sm * 100.0

        # --- Theoretical block occupancy (% of max blocks per SM) ---
        block_occ = (blocks_per_sm / max_blocks_per_sm) * 100.0

        # --- Theoretical warp occupancy ---
        #max_warps_per_sm_thread_limit = max_threads_per_sm // warp_size #just going to define max warps in gpu_config. 
        warps_per_block = wavefronts_per_block
        warps_per_sm = blocks_per_sm * warps_per_block

        warp_occ = (warps_per_sm / self.gpu_config['max_warps_per_sm']) * 100.0
        
        print("thread occupancy: "+str(thread_occ)+"\nblock occupancy: "+str(block_occ)+"\nwarp occupancy: "+str(warp_occ))
        
        occupancy_metrics['Thread Occupancy (%)'] = min(thread_occ, 100.0)
        occupancy_metrics['Block Occupancy (%)'] = min(block_occ, 100.0)
        occupancy_metrics['Warp Occupancy (%)']  = min(warp_occ, 100.0)

        return occupancy_metrics


    def calculate_achieved_occupancy(self):
        """Calculate achieved occupancy from execution statistics"""
        achieved_metrics = {}

        # Look for execution cycles and active cycles
        total_cycles = self._get_total_cycles()
        active_cycles = self._get_active_cycles()
        busy_cycles = self._get_busy_cycles()
        
        
        if total_cycles and active_cycles:
            cycle_occupancy = (active_cycles / total_cycles) * 100
            achieved_metrics['Cycle-based Occupancy (%)'] = cycle_occupancy

        if total_cycles and busy_cycles:
            utilization = (busy_cycles / total_cycles) * 100
            achieved_metrics['SM Utilization (%)'] = utilization
        # Wave/warp occupancy from waveLevelParallelism
        achieved_wave_occ = self._calculate_wave_occupancy()
        if achieved_wave_occ:
            achieved_metrics['Achieved Wave Occupancy (%)'] = achieved_wave_occ
        # Calculate instruction throughput occupancy
        inst_throughput = self._calculate_instruction_throughput()
        if inst_throughput:
            achieved_metrics.update(inst_throughput)

        return achieved_metrics
    
    def _calculate_wave_occupancy(self):
        """Calculate achieved warp/wave occupancy from waveLevelParallelism stats."""
        metrics = {}

        max_waves_per_cu = self.gpu_config["max_warps_per_sm"]  # 40 for gfx9
        num_cus = self.gpu_config["num_sms"]                    # 4 in gpu_config

        active_waves_means = []

        for cu_idx in range(num_cus):
            key = f"system.cpu3.CUs{cu_idx}.waveLevelParallelism::mean"
            entry = self.stats.get(key)
            if entry is None:
                continue

            mean_waves = entry["value"]
            if mean_waves <= 0:
                continue

            active_waves_means.append(mean_waves)

        if not active_waves_means:
            return metrics  

        # Average over CUs
        avg_active_waves = sum(active_waves_means) / len(active_waves_means)

        warp_occ = (avg_active_waves / max_waves_per_cu) * 100.0
        metrics["Achieved Warp Occupancy (%)"] = min(warp_occ, 100.0)

        return metrics["Achieved Warp Occupancy (%)"]

    
    def calculate_resource_occupancy(self):
        """Calculate occupancy limited by various resources"""
        resource_metrics = {}

        # Memory occupancy
        memory_occupancy = self._calculate_memory_occupancy()
        if memory_occupancy:
            resource_metrics.update(memory_occupancy)

        # Register occupancy
        register_occupancy = self._calculate_register_occupancy()
        if register_occupancy:
            resource_metrics.update(register_occupancy)

        # Shared memory occupancy
        shared_mem_occupancy = self._calculate_shared_memory_occupancy()
        if shared_mem_occupancy:
            resource_metrics.update(shared_mem_occupancy)

        return resource_metrics
    
    def _get_active_thread_percent(self):
        active_thread_percent = 0
        for stat_name, stat_data in self.stats.items():
            if any(keyword in stat_name.lower() for keyword in ['valuutilization']):
                active_thread_percent = stat_data['value'] #assume all CUs have same active thread percentage. alternative could be add up all percentages and divide by number of CUs for average active thread percentage
        if active_thread_percent is not None:
            return active_thread_percent
        return None
    
    def _get_active_warps(self):
        """Extract active warps/wavefronts from stats"""
        # Look for wavefront or warp related statistics
        warp_nums = 0
        for stat_name, stat_data in self.stats.items():
            if any(keyword in stat_name.lower() for keyword in ['completedwfs']): #,'wavefront', 'warp', 'simd','wavelevelparallelism'
                #if 'active' in stat_name.lower() or 'launched' in stat_name.lower():
                #print(stat_name.lower())
                #if 'total' in stat_name.lower():
                #print(stat_name)
                #print(stat_data['value'])
                warp_nums += stat_data['value']
        #print(warp_nums)
        if warp_nums is not None:
            return warp_nums
        """
        # Alternative: calculate from thread count
        active_threads = self._get_active_threads()
        if active_threads is not None:
            return active_threads / self.gpu_config['warp_size']
        """
        return None

    def _get_active_threads(self):
        """Extract active thread count"""
        active_threads = 0
        #for stat_name, stat_data in self.stats.items():
        #    if 'thread' in stat_name.lower() and ('active' in stat_name.lower() or 'launched' in stat_name.lower()):
        #        print(stat_data['value'])
        #        busy_cycles += stat_data['value']
        for stat_name, stat_data in self.stats.items():
            if 'valuutilization' in stat_name.lower():
                active_threads += self.gpu_config['warp_size'] * (100 / stat_data['value'])
        if active_threads is not None:
            return active_threads

        # Look for work-item or thread group stats
        for stat_name, stat_data in self.stats.items():
            if 'work_item' in stat_name.lower() or 'workitem' in stat_name.lower():
                return stat_data['value']
        
                
        return None

    def _get_active_blocks(self):
        """Extract active block count"""
        active_blocks = 0
        """
        for stat_name, stat_data in self.stats.items():
            if 'block' in stat_name.lower() and 'active' in stat_name.lower():
                active_blocks += stat_data['value']
        if active_blocks is not None:
            return active_blocks

        # Look for work-group stats (AMD terminology)
        for stat_name, stat_data in self.stats.items():
            if 'work_group' in stat_name.lower() or 'workgroup' in stat_name.lower():
                return stat_data['value']
        """        
        for stat_name, stat_data in self.stats.items():
            if any(keyword in stat_name.lower() for keyword in ['completedwgs']):
                active_blocks += stat_data['value']
        if active_blocks is not None:
            return active_blocks

        return None
    
    def _get_total_wavefronts(self):
        """Sum total completed wavefronts (Wfs) across all CUs"""
        total_wfs = 0
        found = False
        for stat_name, stat_data in self.stats.items():
            #sum over CUs
            if 'completedwfs' in stat_name.lower():
                total_wfs += stat_data['value']
                found = True
        return total_wfs if found else None

    def _get_total_workgroups(self):
        """Sum total completed workgroups (WGs/blocks) across all CUs"""
        total_wgs = 0
        found = False
        for stat_name, stat_data in self.stats.items():
            if 'completedwgs' in stat_name.lower():
                total_wgs += stat_data['value']
                found = True
        return total_wgs if found else None

    
    def _get_total_cycles(self):
        """Total GPU ExecStage cycles across all CUs."""
        total_cycles = 0
        found = False

        for stat_name, stat_data in self.stats.items():
            name = stat_name.lower()
            if ".cus" in name and "execstage.numcycleswithnoissue" in name:
                total_cycles += stat_data["value"]
                found = True
            elif ".cus" in name and "execstage.numcycleswithinstrissued" in name and "::" not in name:
               
                total_cycles += stat_data["value"]
                found = True

        return total_cycles if found else None

    def _get_active_cycles(self):
        """GPU active cycles: cycles where CUs issued at least one instruction."""
        active_cycles = 0
        found = False

        for stat_name, stat_data in self.stats.items():
            name = stat_name.lower()
            if ".cus" in name and "execstage.numcycleswithinstrissued" in name and "::" not in name:
                active_cycles += stat_data["value"]
                found = True

        return active_cycles if found else None
  
    def _get_busy_cycles(self):
        """GPU busy cycles; for ExecStage this is the same as active cycles. This may be differnt in CUDA stats file, but may need to find a different way for ROCm"""
        return self._get_active_cycles()
  
    
    '''
    def _get_busy_cycles(self):
        """Get busy cycles from compute units"""
        total_busy = 0
        found_busy_stats = False

        for stat_name, stat_data in self.stats.items():
            if 'busy' in stat_name.lower() and ('cu' in stat_name.lower() or 'compute_unit' in stat_name.lower()):
                total_busy += stat_data['value']
                found_busy_stats = True

        return total_busy if found_busy_stats else None
    '''
    
    def _calculate_instruction_throughput(self):
        """Calculate instruction-based occupancy metrics using per-CU IPC"""
        metrics = {}

        cu_ipcs = []
        cu_cycles = []

        # assume 4 CUs: CUs0..CUs3
        for cu_idx in range(4):
            ipc_key = f"system.cpu3.CUs{cu_idx}.ipc"
            cycles_key = f"system.cpu3.CUs{cu_idx}.totalCycles"

            ipc_entry = self.stats.get(ipc_key)
            cycles_entry = self.stats.get(cycles_key)

            if not ipc_entry or not cycles_entry:
                continue

            ipc = ipc_entry["value"]
            cycles = cycles_entry["value"]

            # skip NaNs / zero cycles
            if cycles <= 0:
                continue
            if isinstance(ipc, float) and math.isnan(ipc):
                continue

            cu_ipcs.append(ipc)
            cu_cycles.append(cycles)

        if cu_cycles:
            # total instructions across all CUs
            total_insts = sum(ipc * cyc for ipc, cyc in zip(cu_ipcs, cu_cycles))
            print("total insts: "+str(total_insts))
            # total CU-cycles across all CUs
            total_cu_cycles = sum(cu_cycles)
            print("total cu cycles: "+str(total_cu_cycles))

            ipc_avg = total_insts / total_cu_cycles
            metrics["Instructions Per Cycle (avg per CU)"] = ipc_avg

            # IPC-based occupancy vs per-CU theoretical max
            theoretical_max_ipc_per_cu = 1.0  #assumed peak IPC per CU
            ipc_occupancy = (ipc_avg / theoretical_max_ipc_per_cu) * 100.0
            metrics["IPC-based Occupancy (%)"] = min(ipc_occupancy, 100.0)

        return metrics

    '''
    def _calculate_memory_occupancy(self):
        """Calculate memory-related occupancy limitations"""
        metrics = {}

        # Memory bandwidth utilization
        memory_requests = 0
        memory_bytes = 0

        for stat_name, stat_data in self.stats.items():
            if 'mem' in stat_name.lower():
                if 'requests' in stat_name.lower():
                    memory_requests += stat_data['value']
                elif 'bytes' in stat_name.lower():
                    memory_bytes += stat_data['value']

        # Calculate memory bandwidth occupancy
        sim_time = None
        for stat_name, stat_data in self.stats.items():
            if 'simseconds' in stat_name.lower():
                sim_time = stat_data['value']
                break

        if memory_bytes > 0 and sim_time:
            actual_bandwidth = memory_bytes / sim_time / (1024**3)  # GB/s
            # Assume peak memory bandwidth (adjust based on your GPU)
            peak_bandwidth = 900.0  # GB/s for high-end GPU
            bandwidth_occupancy = (actual_bandwidth / peak_bandwidth) * 100
            metrics['Memory Bandwidth Occupancy (%)'] = min(bandwidth_occupancy, 100.0)

        return metrics
    '''

    def _calculate_memory_occupancy(self):
        """
        Calculate memory bandwidth occupancy based on DRAM traffic.

        We use:
          - system.mem_ctrls.dram.bytesRead::total
          - system.mem_ctrls.dram.bytesWritten::total

        to get the total DRAM bytes moved during the simulation, and
        simSeconds for elapsed simulated time.

        Memory Bandwidth Occupancy (%) =
            (Actual DRAM Bandwidth / Peak DRAM Bandwidth) * 100

        Peak bandwidth (GB/s) can be set via:
            self.gpu_config['peak_mem_bandwidth_gbps']
        and defaults to 900.0 GB/s if not provided.
        """
        metrics = {}

        dram_read_bytes = 0
        dram_write_bytes = 0

        # Collect DRAM read/write byte totals
        for stat_name, stat_data in self.stats.items():
            lname = stat_name.lower()
            if "mem_ctrls" in lname and "dram.bytesread::total" in lname:
                dram_read_bytes += stat_data["value"]
            elif "mem_ctrls" in lname and "dram.byteswritten::total" in lname:
                dram_write_bytes += stat_data["value"]
        
        
        memory_bytes = dram_read_bytes + dram_write_bytes
        print("memory_bytes: "+str(memory_bytes))
        if memory_bytes <= 0:
            return metrics  # no DRAM usage recorded

        # Get sim time in seconds
        sim_time = None
        for stat_name, stat_data in self.stats.items():
            lname = stat_name.lower()
            if "simseconds" in lname:
                sim_time = stat_data["value"]
                break

        if sim_time is None or sim_time <= 0:
            return metrics  # can't compute bandwidth without time
        print(memory_bytes)
        # Actual DRAM bandwidth in GB/s
        actual_bandwidth_gbps = memory_bytes / sim_time / (1024 ** 3)

        # Peak bandwidth is a config knob
        peak_bandwidth_gbps = self.gpu_config.get("peak_mem_bandwidth_gbps", 900.0) #changed from 900 due to simulated hardware

        if peak_bandwidth_gbps and peak_bandwidth_gbps > 0.0:
            bw_occ = (actual_bandwidth_gbps / peak_bandwidth_gbps) * 100.0
            bw_occ = min(bw_occ, 100.0)
            print("memory bw occ: "+str(bw_occ))
            metrics["Actual DRAM Bandwidth (GB/s)"] = actual_bandwidth_gbps
            metrics["Memory Bandwidth Occupancy (%)"] = bw_occ

        return metrics


    def _calculate_register_occupancy(self):
        """Calculate register usage occupancy"""
        metrics = {}

        # Look for register usage statistics
        for stat_name, stat_data in self.stats.items():
            if 'register' in stat_name.lower() or 'reg' in stat_name.lower():
                if 'usage' in stat_name.lower() or 'used' in stat_name.lower():
                    reg_usage = stat_data['value']
                    reg_occupancy = (reg_usage / self.gpu_config['registers_per_sm']) * 100
                    metrics['Register Occupancy (%)'] = min(reg_occupancy, 100.0)

        return metrics

    def _calculate_shared_memory_occupancy(self):
        """Calculate shared memory occupancy"""
        metrics = {}

        # Look for shared memory or LDS (Local Data Share) usage
        for stat_name, stat_data in self.stats.items():
            if any(keyword in stat_name.lower() for keyword in ['shared_mem', 'lds', 'local_mem']):
                if 'usage' in stat_name.lower() or 'used' in stat_name.lower():
                    shared_usage = stat_data['value']
                    shared_occupancy = (shared_usage / self.gpu_config['shared_memory_per_sm']) * 100
                    metrics['Shared Memory Occupancy (%)'] = min(shared_occupancy, 100.0)

        return metrics

    def analyze_occupancy(self):
        """Perform comprehensive occupancy analysis"""
        print("\n🔍 ANALYZING GPU OCCUPANCY")
        print("=" * 50)

        # Calculate different types of occupancy
        theoretical = self.calculate_theoretical_occupancy()
        achieved = self.calculate_achieved_occupancy()
        resource_limited = self.calculate_resource_occupancy()

        self.occupancy_data = {
            'Theoretical Occupancy': theoretical,
            'Achieved Occupancy': achieved,
            'Resource-Limited Occupancy': resource_limited
        }

        return self.occupancy_data

    def print_occupancy_report(self):
        """Print detailed occupancy report"""
        print("\n📊 GPU OCCUPANCY ANALYSIS REPORT")
        print("=" * 60)

        for category, metrics in self.occupancy_data.items():
            if metrics:
                print(f"\n🎯 {category.upper()}:")
                print("-" * 40)

                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        if '%' in metric_name:
                            print(f"  {metric_name:<35}: {value:6.2f}%")
                        else:
                            print(f"  {metric_name:<35}: {value:8.4f}")
                    else:
                        print(f"  {metric_name:<35}: {value:,}")

        # Calculate overall occupancy estimate
        self._print_occupancy_summary()

    def _print_occupancy_summary(self):
        """Print occupancy summary and recommendations"""
        print(f"\n🎯 OCCUPANCY SUMMARY")
        print("=" * 40)

        # Find the limiting factor
        all_occupancies = []
        limiting_factors = []

        for category, metrics in self.occupancy_data.items():
            for metric_name, value in metrics.items():
                if '%' in metric_name and isinstance(value, (int, float)):
                    all_occupancies.append((metric_name, value))
                    if value < 50:  # Consider low occupancy
                        limiting_factors.append((metric_name, value))

        if all_occupancies:
            # Find average occupancy
            avg_occupancy = sum(occ[1] for occ in all_occupancies) / len(all_occupancies)
            print(f"Average Occupancy: {avg_occupancy:.2f}%")

            # Find minimum occupancy (bottleneck)
            min_occupancy = min(all_occupancies, key=lambda x: x[1])
            print(f"Bottleneck: {min_occupancy[0]} ({min_occupancy[1]:.2f}%)")

        if limiting_factors:
            print(f"\n⚠️  LOW OCCUPANCY FACTORS:")
            for factor, value in limiting_factors:
                print(f"   • {factor}: {value:.2f}%")

        # Recommendations
        self._print_recommendations()

    def _print_recommendations(self):
        """Print optimization recommendations based on occupancy analysis"""
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)

        recommendations = []

        # Check various occupancy metrics for recommendations
        for category, metrics in self.occupancy_data.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and '%' in metric_name:
                    if 'Memory' in metric_name and value < 50:
                        recommendations.append("• Optimize memory access patterns (coalescing)")
                        recommendations.append("• Consider using shared memory to reduce global memory traffic")
                    elif 'Register' in metric_name and value > 80:
                        recommendations.append("• Reduce register usage per thread")
                        recommendations.append("• Consider spilling some variables to shared memory")
                    elif 'Warp' in metric_name and value < 60:
                        recommendations.append("• Increase threads per block")
                        recommendations.append("• Optimize thread divergence")
                    elif 'SM Utilization' in metric_name and value < 70:
                        recommendations.append("• Increase number of blocks per kernel launch")
                        recommendations.append("• Balance work distribution across SMs")

        # Remove duplicates and print
        unique_recommendations = list(set(recommendations))
        if unique_recommendations:
            for rec in unique_recommendations[:5]:  # Show top 5 recommendations
                print(f"  {rec}")
        else:
            print("  • Occupancy appears to be well optimized!")

    def create_occupancy_visualization(self):
        """Create occupancy visualization graphs"""
        if not self.occupancy_data:
            print("No occupancy data to visualize")
            return

        # Collect all percentage metrics
        occupancy_metrics = {}
        for category, metrics in self.occupancy_data.items():
            for metric_name, value in metrics.items():
                if '%' in metric_name and isinstance(value, (int, float)):
                    occupancy_metrics[metric_name.replace(' (%)', '')] = value

        if not occupancy_metrics:
            print("No percentage-based occupancy metrics found")
            return

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar chart of occupancy metrics
        metrics_names = list(occupancy_metrics.keys())
        metrics_values = list(occupancy_metrics.values())

        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_names)))
        bars = ax1.bar(range(len(metrics_names)), metrics_values, color=colors)

        ax1.set_title('GPU Occupancy Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Occupancy (%)')
        ax1.set_ylim(0, 100)
        ax1.set_xticks(range(len(metrics_names)))
        ax1.set_xticklabels(metrics_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            if height > 95:
                y = height - 5
                color = 'white'
            else:
                y = height + 1
                color = 'black'
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f'{value:.1f}%',
                ha='center',
                va='bottom',
                color=color,
                fontsize=9
            )

        # Add horizontal lines for reference
        ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% threshold')
        ax1.axhline(y=75, color='green', linestyle='--', alpha=0.7, label='75% target')
        ax1.legend()

        # Pie chart of occupancy distribution
        if len(metrics_values) > 1:
            normalized_values = [max(v, 1e-3) for v in metrics_values]

            def autopct_hide_small(pct):
                return f'{pct:.1f}%' if pct >= 3 else ''

            wedges, texts, autotexts = ax2.pie(
                normalized_values,
                labels=metrics_names,
                autopct=autopct_hide_small,
                startangle=90,
                pctdistance=0.75
            )

            # Make pie look nicer
            ax2.axis('equal')  # Equal aspect ratio -> circle

            # adjusting font sizes so labels don't overlap as much
            for t in texts:
                t.set_fontsize(8)
            for t in autotexts:
                t.set_fontsize(8)

            ax2.set_title('Occupancy Distribution', fontsize=14, fontweight='bold')
        else:
            ax2.text(
                0.5, 0.5,
                'Insufficient data\nfor pie chart',
                ha='center', va='center', transform=ax2.transAxes
            )
            ax2.set_title('Occupancy Distribution', fontsize=14, fontweight='bold')


        plt.tight_layout()
        plt.savefig('gpu_occupancy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Occupancy visualization saved as: gpu_occupancy_analysis.png")

    def save_occupancy_data(self, output_file):
        """Save occupancy analysis to JSON file"""
        try:
            # Add GPU configuration to output
            output_data = {
                'gpu_configuration': self.gpu_config,
                'occupancy_analysis': self.occupancy_data,
                'analysis_metadata': {
                    'stats_file': self.stats_file,
                    'total_stats_parsed': len(self.stats)
                }
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"💾 Occupancy analysis saved to: {output_file}")
        except Exception as e:
            print(f"Error saving occupancy data: {e}")

def main():
    # Default path
    default_stats_path = "/workspaces/intro-to-gem5-alex-keist/m5out/stats.txt"

    if len(sys.argv) > 1:
        stats_path = sys.argv[1]
    else:
        stats_path = default_stats_path

    print(f"🔍 Analyzing GPU occupancy from: {stats_path}")

    if not os.path.exists(stats_path):
        print(f"❌ Stats file not found at: {stats_path}")
        sys.exit(1)

    # Create analyzer
    analyzer = GPUOccupancyAnalyzer(stats_path)

    # Parse stats
    if not analyzer.parse_stats_file():
        sys.exit(1)

    # Analyze occupancy
    analyzer.analyze_occupancy()

    # Print report
    analyzer.print_occupancy_report()

    # Create visualization
    analyzer.create_occupancy_visualization()

    # Save results
    output_dir = os.path.dirname(stats_path)
    json_output = os.path.join(output_dir, "gpu_occupancy_analysis.json")
    analyzer.save_occupancy_data(json_output)

    print(f"\n✅ GPU occupancy analysis complete!")

if __name__ == "__main__":
    main()
