#!/usr/bin/env python3
"""
Carbonyl Orientation Analysis Module for K+ Channel Selectivity Filter

This module tracks the rotation/orientation of carbonyl groups in the K+ channel
selectivity filter, particularly focusing on the TVGYG sequence that coordinates
potassium ions. It calculates relevant dihedral angles and analyzes their
distributions and time evolution in toxin-bound versus control systems.

Key features:
- Identifies selectivity filter residues in K+ channel
- Calculates backbone dihedral angles (phi/psi) that affect carbonyl orientation
- Specifically tracks the C-N-CA-C (phi) and N-CA-C-N (psi) dihedrals
- Analyzes carbonyl flipping events and stability
- Correlates carbonyl orientation with ion occupancy

This can be used as a standalone script or integrated into the main analysis suite.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import argparse
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.analysis.dihedrals import Ramachandran
import warnings

# Suppress MDAnalysis warnings that might flood output
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

class CarbonylAnalyzer:
    """
    Analyzes carbonyl group orientations in K+ channel selectivity filter.
    """
    
    def __init__(self, psf_file, dcd_file, output_dir, system_type="unknown"):
        """
        Initialize the analyzer with trajectory files.
        
        Parameters:
        -----------
        psf_file : str
            Path to the PSF topology file
        dcd_file : str
            Path to the DCD trajectory file
        output_dir : str
            Directory to save analysis results
        system_type : str
            'toxin' or 'control' to categorize the system
        """
        self.psf_file = psf_file
        self.dcd_file = dcd_file
        self.output_dir = output_dir
        self.system_type = system_type.lower()
        self.run_name = os.path.basename(os.path.dirname(dcd_file))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the trajectory
        try:
            self.u = mda.Universe(psf_file, dcd_file)
            self.n_frames = len(self.u.trajectory)
            print(f"Loaded trajectory with {self.n_frames} frames")
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            raise
            
        # Initialize data storage
        self.filter_residues = {}  # Will store filter residues by chain
        self.dihedral_data = {}    # Will store dihedral angle data
        self.carbonyl_flips = {}   # Will store carbonyl flip events
        
    def identify_filter_residues(self):
        """
        Identify the selectivity filter residues (TVGYG) in each channel subunit.
        """
        print("Identifying selectivity filter residues...")
        
        # Find channel subunits (chains)
        channel_chains = []
        potential_segids = ['PROA', 'PROB', 'PROC', 'PROD', 'A', 'B', 'C', 'D']
        
        for segid in potential_segids:
            seg_atoms = self.u.select_atoms(f'segid {segid}')
            if len(seg_atoms) > 0:
                channel_chains.append(segid)
        
        if not channel_chains:
            raise ValueError("No channel chains found. Check segids in the PSF file.")
        
        print(f"Found channel chains: {channel_chains}")
        
        # For each chain, identify the selectivity filter residues
        # The classic K+ channel selectivity filter sequence is TVGYG
        # We'll search for the GYG motif and include the two residues before it
        for chain in channel_chains:
            chain_atoms = self.u.select_atoms(f'segid {chain}')
            chain_residues = chain_atoms.residues
            
            # Convert three-letter codes to one-letter
            aa_codes = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                       'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                       'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                       'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                       'HSE': 'H', 'HSD': 'H', 'HSP': 'H'}  # Handle CHARMM histidine variants
            
            # Build sequence
            seq = ""
            resids = []
            for res in chain_residues:
                if res.resname in aa_codes:
                    seq += aa_codes[res.resname]
                    resids.append(res.resid)
                else:
                    seq += 'X'  # Unknown residue
                    resids.append(res.resid)
            
            # Find the GYG motif
            gyg_idx = seq.rfind('GYG')  # Find the last occurrence
            
            if gyg_idx != -1 and gyg_idx >= 2:
                # Found GYG, extract the TVGYG equivalent (might not actually be TVGYG)
                filter_start = gyg_idx - 2
                filter_end = gyg_idx + 3  # Non-inclusive, so this gives 5 residues
                
                # Get the residue IDs for these positions
                filter_resids = resids[filter_start:filter_end]
                filter_seq = seq[filter_start:filter_end]
                
                # Store the filter residues for this chain
                self.filter_residues[chain] = {
                    'resids': filter_resids,
                    'sequence': filter_seq
                }
                
                print(f"Chain {chain}: Found filter sequence {filter_seq} with resids {filter_resids}")
            else:
                print(f"Warning: Could not find GYG motif in chain {chain}")
        
        # Check if we found filter residues
        if not self.filter_residues:
            raise ValueError("Failed to identify selectivity filter residues in any chain")
            
        return self.filter_residues
    
    def setup_dihedral_selections(self):
        """
        Set up atom selections for calculating dihedrals affecting carbonyl orientation.
        
        For each residue in the filter, we'll track:
        1. Phi (C-N-CA-C): affects carbonyl orientation
        2. Psi (N-CA-C-N): directly relates to carbonyl orientation
        3. Omega (CA-C-N-CA): peptide bond planarity (should be ~180°)
        """
        print("Setting up dihedral angle selections...")
        
        # Dictionary to store atom selections for each dihedral
        dihedral_selections = {}
        
        # Go through each chain and its filter residues
        for chain, filter_info in self.filter_residues.items():
            chain_dihedrals = {}
            
            # Process each residue in the filter
            for i, resid in enumerate(filter_info['resids']):
                # Extract the residue letter (e.g., T, V, G, Y, G)
                res_letter = filter_info['sequence'][i]
                # Residue position name in the filter (e.g., T, V, G1, Y, G2)
                if res_letter == 'G' and i > 0 and filter_info['sequence'][i-1] == 'Y':
                    position_name = 'G2'  # Second glycine
                elif res_letter == 'G' and i > 0:
                    position_name = 'G1'  # First glycine
                else:
                    position_name = res_letter
                
                # Need to handle the first residue (can't calculate phi) and last residue (can't calculate psi)
                if i > 0:  # Not the first residue, can calculate phi
                    prev_resid = filter_info['resids'][i-1]
                    # Atoms for phi: C(i-1), N(i), CA(i), C(i)
                    phi_atoms = self.u.select_atoms(
                        f"segid {chain} and resid {prev_resid} and name C",  # C of previous residue
                        f"segid {chain} and resid {resid} and name N",       # N of current residue
                        f"segid {chain} and resid {resid} and name CA",      # CA of current residue
                        f"segid {chain} and resid {resid} and name C"        # C of current residue
                    )
                    
                    if len(phi_atoms) == 4:
                        chain_dihedrals[f"{position_name}_phi"] = phi_atoms
                    else:
                        print(f"Warning: Could not select all atoms for phi of {chain}:{position_name} (resid {resid})")
                
                if i < len(filter_info['resids']) - 1:  # Not the last residue, can calculate psi
                    next_resid = filter_info['resids'][i+1]
                    # Atoms for psi: N(i), CA(i), C(i), N(i+1)
                    psi_atoms = self.u.select_atoms(
                        f"segid {chain} and resid {resid} and name N",        # N of current residue
                        f"segid {chain} and resid {resid} and name CA",       # CA of current residue
                        f"segid {chain} and resid {resid} and name C",        # C of current residue
                        f"segid {chain} and resid {next_resid} and name N"    # N of next residue
                    )
                    
                    if len(psi_atoms) == 4:
                        chain_dihedrals[f"{position_name}_psi"] = psi_atoms
                    else:
                        print(f"Warning: Could not select all atoms for psi of {chain}:{position_name} (resid {resid})")
                
                # Carbonyl specific dihedral (O-C-CA-N)
                # This dihedral specifically reflects the carbonyl orientation
                carbonyl_atoms = self.u.select_atoms(
                    f"segid {chain} and resid {resid} and name O",       # Carbonyl O
                    f"segid {chain} and resid {resid} and name C",       # Carbonyl C
                    f"segid {chain} and resid {resid} and name CA",      # Alpha carbon
                    f"segid {chain} and resid {resid} and name N"        # Backbone N
                )
                
                if len(carbonyl_atoms) == 4:
                    chain_dihedrals[f"{position_name}_carbonyl"] = carbonyl_atoms
                else:
                    print(f"Warning: Could not select all atoms for carbonyl dihedral of {chain}:{position_name} (resid {resid})")
            
            # Store the dihedrals for this chain
            dihedral_selections[chain] = chain_dihedrals
        
        return dihedral_selections
    
    def calculate_dihedrals(self):
        """
        Calculate dihedral angles for all frames in the trajectory.
        """
        print("Calculating dihedral angles...")
        
        # Set up dihedral selections if not already done
        if not hasattr(self, 'dihedral_selections'):
            self.dihedral_selections = self.setup_dihedral_selections()
        
        # Dictionary to store dihedral angles
        dihedral_data = {}
        
        # Setup progress bar
        pbar = tqdm(total=self.n_frames * sum(len(dihedrals) for dihedrals in self.dihedral_selections.values()),
                   desc="Calculating dihedrals")
        
        # Iterate through trajectory
        for ts in self.u.trajectory:
            frame_num = ts.frame
            
            # Create entry for this frame
            if frame_num not in dihedral_data:
                dihedral_data[frame_num] = {}
            
            # Calculate dihedrals for each chain
            for chain, dihedrals in self.dihedral_selections.items():
                # Create entry for this chain
                if chain not in dihedral_data[frame_num]:
                    dihedral_data[frame_num][chain] = {}
                
                # Calculate each dihedral
                for dihedral_name, atoms in dihedrals.items():
                    # Calculate dihedral angle
                    angle = mda.lib.distances.calc_dihedrals(
                        atoms[0].position,
                        atoms[1].position,
                        atoms[2].position,
                        atoms[3].position
                    )
                    
                    # Convert to degrees and store
                    angle_deg = np.rad2deg(angle)
                    dihedral_data[frame_num][chain][dihedral_name] = angle_deg
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Reorganize data for easier analysis
        # From: {frame: {chain: {dihedral: angle}}} 
        # To: {chain: {dihedral: [angles for each frame]}}
        reorganized_data = {}
        
        for chain in self.dihedral_selections.keys():
            reorganized_data[chain] = {}
            
            # Get all dihedral names for this chain
            dihedral_names = set()
            for frame_data in dihedral_data.values():
                if chain in frame_data:
                    dihedral_names.update(frame_data[chain].keys())
            
            # Initialize arrays for each dihedral
            for dihedral_name in dihedral_names:
                reorganized_data[chain][dihedral_name] = []
            
            # Fill in the data
            for frame_num in sorted(dihedral_data.keys()):
                if chain in dihedral_data[frame_num]:
                    for dihedral_name in dihedral_names:
                        if dihedral_name in dihedral_data[frame_num][chain]:
                            reorganized_data[chain][dihedral_name].append(
                                dihedral_data[frame_num][chain][dihedral_name]
                            )
                        else:
                            # Missing data for this frame
                            reorganized_data[chain][dihedral_name].append(np.nan)
        
        self.dihedral_data = reorganized_data
        
        # Generate time points (assuming frames are evenly spaced)
        self.time_points = np.arange(self.n_frames) * self.u.trajectory.dt / 1000  # Convert to ns
        
        return self.dihedral_data
    
    def detect_carbonyl_flips(self, threshold=45.0):
        """
        Detect carbonyl flipping events based on changes in dihedral angles.
        
        Parameters:
        -----------
        threshold : float
            Minimum change in degrees to consider a flip event
        """
        print(f"Detecting carbonyl flips (threshold: {threshold}°)...")
        
        carbonyl_flips = {}
        
        # Iterate through chains and dihedrals
        for chain, dihedrals in self.dihedral_data.items():
            chain_flips = {}
            
            # Focus on carbonyl-specific dihedrals
            for dihedral_name, angles in dihedrals.items():
                if 'carbonyl' in dihedral_name or 'psi' in dihedral_name:
                    angles_array = np.array(angles)
                    
                    # Calculate differences between consecutive frames
                    # Handle angle wrapping (e.g., -175° to 175° should be a 10° change, not 350°)
                    angle_diffs = []
                    for i in range(1, len(angles_array)):
                        diff = angles_array[i] - angles_array[i-1]
                        # Adjust for circular nature of angles
                        if diff > 180:
                            diff -= 360
                        elif diff < -180:
                            diff += 360
                        angle_diffs.append(abs(diff))
                    
                    # Find flip events (large changes in angle)
                    flip_indices = [i for i, diff in enumerate(angle_diffs) if diff > threshold]
                    
                    # Store flip events
                    if flip_indices:
                        # Convert to frame numbers (add 1 because diff is between frames)
                        flip_frames = [i + 1 for i in flip_indices]
                        
                        # Store time points, angle before, and angle after for each flip
                        flip_details = []
                        for frame in flip_frames:
                            flip_details.append({
                                'frame': frame,
                                'time': self.time_points[frame],
                                'angle_before': angles_array[frame-1],
                                'angle_after': angles_array[frame],
                                'change': angle_diffs[frame-1]
                            })
                        
                        chain_flips[dihedral_name] = flip_details
            
            carbonyl_flips[chain] = chain_flips
        
        self.carbonyl_flips = carbonyl_flips
        
        # Count total flips
        total_flips = sum(len(flips) for chain_flips in carbonyl_flips.values() 
                         for flips in chain_flips.values())
        
        print(f"Detected {total_flips} carbonyl flip events across all chains")
        
        return carbonyl_flips
    
    def save_dihedral_data(self):
        """
        Save dihedral angle data to CSV files.
        """
        print("Saving dihedral data to CSV files...")
        
        # Create a directory for dihedral data
        dihedral_dir = os.path.join(self.output_dir, 'dihedral_data')
        os.makedirs(dihedral_dir, exist_ok=True)
        
        # Save overall dihedral data
        for chain, dihedrals in self.dihedral_data.items():
            # Create DataFrame
            data = {'Time (ns)': self.time_points}
            
            # Add each dihedral angle series
            for dihedral_name, angles in dihedrals.items():
                data[f"{dihedral_name}"] = angles
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            filepath = os.path.join(dihedral_dir, f"{chain}_dihedrals.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {filepath}")
        
        # Save carbonyl flip events
        if hasattr(self, 'carbonyl_flips'):
            all_flips = []
            
            for chain, chain_flips in self.carbonyl_flips.items():
                for dihedral_name, flips in chain_flips.items():
                    for flip in flips:
                        flip_data = {
                            'Chain': chain,
                            'Dihedral': dihedral_name,
                            'Frame': flip['frame'],
                            'Time (ns)': flip['time'],
                            'Angle Before (deg)': flip['angle_before'],
                            'Angle After (deg)': flip['angle_after'],
                            'Change (deg)': flip['change']
                        }
                        all_flips.append(flip_data)
            
            if all_flips:
                # Create DataFrame
                flips_df = pd.DataFrame(all_flips)
                
                # Save to CSV
                filepath = os.path.join(dihedral_dir, "carbonyl_flips.csv")
                flips_df.to_csv(filepath, index=False)
                print(f"Saved {filepath}")
            else:
                print("No carbonyl flips detected to save")
    
    def plot_dihedral_timeseries(self):
        """
        Create time series plots of dihedral angles for each chain and residue.
        """
        print("Creating dihedral time series plots...")
        
        # Create a directory for plots
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot settings
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Iterate through chains
        for chain, dihedrals in self.dihedral_data.items():
            # Group dihedrals by residue position
            residue_dihedrals = {}
            
            for dihedral_name in dihedrals.keys():
                # Extract residue position (e.g., T, V, G1, Y, G2)
                residue_pos = dihedral_name.split('_')[0]
                dihedral_type = dihedral_name.split('_')[1]
                
                if residue_pos not in residue_dihedrals:
                    residue_dihedrals[residue_pos] = []
                
                residue_dihedrals[residue_pos].append((dihedral_name, dihedral_type))
            
            # Create a plot for each residue position
            for residue_pos, dihedral_info in residue_dihedrals.items():
                # Skip if no dihedrals for this residue
                if not dihedral_info:
                    continue
                
                # Create plot
                fig, ax = plt.subplots()
                
                # Plot each dihedral
                for dihedral_name, dihedral_type in dihedral_info:
                    angles = dihedrals[dihedral_name]
                    
                    # Set line properties based on dihedral type
                    if dihedral_type == 'phi':
                        color = 'blue'
                        label = 'Phi (C-N-CA-C)'
                    elif dihedral_type == 'psi':
                        color = 'red'
                        label = 'Psi (N-CA-C-N)'
                    elif dihedral_type == 'carbonyl':
                        color = 'green'
                        label = 'Carbonyl (O-C-CA-N)'
                    else:
                        color = 'gray'
                        label = dihedral_type
                    
                    # Plot the data
                    ax.plot(self.time_points, angles, color=color, label=label)
                
                # Add carbonyl flip markers if detected
                if hasattr(self, 'carbonyl_flips') and chain in self.carbonyl_flips:
                    chain_flips = self.carbonyl_flips[chain]
                    
                    for dihedral_name in dihedrals.keys():
                        if dihedral_name in chain_flips:
                            flips = chain_flips[dihedral_name]
                            
                            # Plot markers for flip events
                            for flip in flips:
                                ax.axvline(x=flip['time'], color='black', linestyle='--', alpha=0.5)
                                ax.scatter(flip['time'], flip['angle_after'], color='purple', marker='o')
                
                # Add horizontal line at 0°
                ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
                
                # Set title and labels
                system_label = f"Toxin" if self.system_type == 'toxin' else f"Control"
                ax.set_title(f"{system_label} System - Chain {chain} - {residue_pos} Dihedral Angles")
                ax.set_xlabel("Time (ns)")
                ax.set_ylabel("Dihedral Angle (degrees)")
                
                # Set y-limits to focus on relevant angle range
                ax.set_ylim(-180, 180)
                
                # Add legend
                ax.legend()
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Save the figure
                plt.tight_layout()
                filepath = os.path.join(plot_dir, f"{chain}_{residue_pos}_dihedrals.png")
                plt.savefig(filepath, dpi=150)
                plt.close()
    
    def plot_carbonyl_orientation_summary(self):
        """
        Create summary plots of carbonyl orientations and flipping statistics.
        """
        print("Creating carbonyl orientation summary plots...")
        
        # Create a directory for plots
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot settings
        plt.rcParams['figure.figsize'] = (12, 10)
        
        # 1. Create a combined plot with all carbonyl orientations
        fig, axes = plt.subplots(nrows=len(self.dihedral_data), ncols=1, 
                               sharex=True, figsize=(12, 3*len(self.dihedral_data)))
        
        # Handle single chain case
        if len(self.dihedral_data) == 1:
            axes = [axes]
        
        for ax_idx, (chain, dihedrals) in enumerate(self.dihedral_data.items()):
            ax = axes[ax_idx]
            
            # Plot each carbonyl dihedral
            for dihedral_name, angles in dihedrals.items():
                if 'carbonyl' in dihedral_name:
                    # Extract residue position
                    residue_pos = dihedral_name.split('_')[0]
                    
                    # Plot the data
                    ax.plot(self.time_points, angles, label=f"{residue_pos}")
            
            # Add horizontal line at 0°
            ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
            
            # Set title and labels
            ax.set_title(f"Chain {chain} - Carbonyl Orientations")
            ax.set_ylabel("Dihedral Angle (degrees)")
            
            # Set y-limits to focus on relevant angle range
            ax.set_ylim(-180, 180)
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        # Set common x-label
        fig.text(0.5, 0.04, "Time (ns)", ha='center', va='center', fontsize=12)
        
        # Save the figure
        plt.tight_layout()
        filepath = os.path.join(plot_dir, "all_carbonyl_orientations.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        # 2. Create a heatmap of carbonyl angles over time
        if self.n_frames > 1:  # Only create heatmap if we have multiple frames
            for chain, dihedrals in self.dihedral_data.items():
                # Collect carbonyl dihedrals
                carbonyl_data = {}
                for dihedral_name, angles in dihedrals.items():
                    if 'carbonyl' in dihedral_name:
                        residue_pos = dihedral_name.split('_')[0]
                        carbonyl_data[residue_pos] = angles
                
                # Skip if no carbonyl dihedrals
                if not carbonyl_data:
                    continue
                
                # Create DataFrame
                df = pd.DataFrame(carbonyl_data)
                
                # Create heatmap
                plt.figure(figsize=(12, 8))
                
                # Determine appropriate time sampling for readability
                time_step = max(1, self.n_frames // 100)  # Show at most 100 time points
                sampled_times = self.time_points[::time_step]
                sampled_data = df.iloc[::time_step]
                
                # Create heatmap
                ax = sns.heatmap(sampled_data.T, cmap='coolwarm', vmin=-180, vmax=180,
                               cbar_kws={'label': 'Dihedral Angle (degrees)'})
                
                # Set title and labels
                system_label = f"Toxin" if self.system_type == 'toxin' else f"Control"
                plt.title(f"{system_label} System - Chain {chain} - Carbonyl Angle Heatmap")
                plt.xlabel("Frame Index")
                plt.ylabel("Residue Position")
                
                # Set x-ticks to show time in ns
                ax.set_xticks(range(0, len(sampled_times), max(1, len(sampled_times) // 10)))
                ax.set_xticklabels([f"{t:.1f}" for t in sampled_times[::max(1, len(sampled_times) // 10)]])
                
                # Save the figure
                plt.tight_layout()
                filepath = os.path.join(plot_dir, f"{chain}_carbonyl_heatmap.png")
                plt.savefig(filepath, dpi=150)
                plt.close()
        
        # 3. Create a histogram of carbonyl angles
        for chain, dihedrals in self.dihedral_data.items():
            # Collect carbonyl dihedrals
            carbonyl_data = {}
            for dihedral_name, angles in dihedrals.items():
                if 'carbonyl' in dihedral_name:
                    residue_pos = dihedral_name.split('_')[0]
                    carbonyl_data[residue_pos] = angles
            
            # Skip if no carbonyl dihedrals
            if not carbonyl_data:
                continue
            
            # Create plot
            fig, ax = plt.subplots()
            
            # Plot histogram for each residue
            for residue_pos, angles in carbonyl_data.items():
                sns.histplot(angles, bins=36, alpha=0.6, label=residue_pos, ax=ax)
            
            # Set title and labels
            system_label = f"Toxin" if self.system_type == 'toxin' else f"Control"
            ax.set_title(f"{system_label} System - Chain {chain} - Carbonyl Angle Distribution")
            ax.set_xlabel("Dihedral Angle (degrees)")
            ax.set_ylabel("Frequency")
            
            # Set x-limits to show full range
            ax.set_xlim(-180, 180)
            
            # Add legend
            ax.legend()
            
            # Save the figure
            plt.tight_layout()
            filepath = os.path.join(plot_dir, f"{chain}_carbonyl_histogram.png")
            plt.savefig(filepath, dpi=150)
            plt.close()
        
        # 4. Create flip statistics plot (if flips detected)
        if hasattr(self, 'carbonyl_flips'):
            # Count flips per residue and chain
            flip_counts = {}
            
            for chain, chain_flips in self.carbonyl_flips.items():
                chain_counts = {}
                
                for dihedral_name, flips in chain_flips.items():
                    residue_pos = dihedral_name.split('_')[0]
                    dihedral_type = dihedral_name.split('_')[1]
                    
                    key = f"{residue_pos}_{dihedral_type}"
                    chain_counts[key] = len(flips)
                
                flip_counts[chain] = chain_counts
            
            # Create bar plot
            if flip_counts:
                # Combine data across chains
                all_counts = {}
                
                for chain, counts in flip_counts.items():
                    for key, count in counts.items():
                        if key not in all_counts:
                            all_counts[key] = {}
                        all_counts[key][chain] = count
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Set position for each group of bars
                positions = np.arange(len(all_counts))
                width = 0.8 / len(self.dihedral_data)
                
                # Plot bars for each chain
                for i, chain in enumerate(self.dihedral_data.keys()):
                    counts = [all_counts[key].get(chain, 0) for key in all_counts.keys()]
                    pos = positions + i * width
                    
                    ax.bar(pos, counts, width=width, label=f"Chain {chain}")
                
                # Set title and labels
                system_label = f"Toxin" if self.system_type == 'toxin' else f"Control"
                ax.set_title(f"{system_label} System - Carbonyl Flip Events")
                ax.set_xlabel("Residue and Dihedral Type")
                ax.set_ylabel("Number of Flips")
                
                # Set x-ticks
                ax.set_xticks(positions + width * (len(self.dihedral_data) - 1) / 2)
                ax.set_xticklabels(list(all_counts.keys()), rotation=45, ha='right')
                
                # Add legend
                ax.legend()
                
                # Save the figure
                plt.tight_layout()
                filepath = os.path.join(plot_dir, "carbonyl_flip_statistics.png")
                plt.savefig(filepath, dpi=150)
                plt.close()
    
    def run_analysis(self):
        """
        Run the complete carbonyl analysis workflow.
        """
        print(f"Running carbonyl analysis for {self.run_name} ({self.system_type} system)...")
        
        # 1. Identify filter residues
        self.identify_filter_residues()
        
        # 2. Setup dihedral selections
        self.dihedral_selections = self.setup_dihedral_selections()
        
        # 3. Calculate dihedrals
        self.calculate_dihedrals()
        
        # 4. Detect carbonyl flips
        self.detect_carbonyl_flips()
        
        # 5. Save data
        self.save_dihedral_data()
        
        # 6. Create plots
        self.plot_dihedral_timeseries()
        self.plot_carbonyl_orientation_summary()
        
        print(f"Carbonyl analysis completed for {self.run_name}.")
        return self.dihedral_data, self.carbonyl_flips


def compare_carbonyl_orientations(toxin_results, control_results, output_dir):
    """
    Compare carbonyl orientations between toxin and control systems.
    
    Parameters:
    -----------
    toxin_results : list of tuples
        List of (dihedral_data, carbonyl_flips) for toxin systems
    control_results : list of tuples
        List of (dihedral_data, carbonyl_flips) for control systems
    output_dir : str
        Directory to save comparison results
    """
    print("Comparing carbonyl orientations between toxin and control systems...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract dihedral data from results
    toxin_dihedrals = [result[0] for result in toxin_results]
    control_dihedrals = [result[0] for result in control_results]
    
    # Extract flip data from results
    toxin_flips = [result[1] for result in toxin_results]
    control_flips = [result[1] for result in control_results]
    
    # Collect statistics per residue and dihedral type
    stats = {}
    
    # Process each chain separately
    all_chains = set()
    for dihedrals in toxin_dihedrals + control_dihedrals:
        all_chains.update(dihedrals.keys())
    
    for chain in all_chains:
        chain_stats = {}
        
        # Find all dihedral types for this chain
        dihedral_types = set()
        for dihedrals in toxin_dihedrals + control_dihedrals:
            if chain in dihedrals:
                dihedral_types.update(dihedrals[chain].keys())
        
        for dihedral_type in dihedral_types:
            # Collect toxin data for this dihedral
            toxin_angles = []
            for dihedrals in toxin_dihedrals:
                if chain in dihedrals and dihedral_type in dihedrals[chain]:
                    toxin_angles.extend(dihedrals[chain][dihedral_type])
            
            # Collect control data for this dihedral
            control_angles = []
            for dihedrals in control_dihedrals:
                if chain in dihedrals and dihedral_type in dihedrals[chain]:
                    control_angles.extend(dihedrals[chain][dihedral_type])
            
            # Calculate statistics if we have data
            if toxin_angles and control_angles:
                # Calculate mean and standard deviation
                toxin_mean = np.nanmean(toxin_angles)
                toxin_std = np.nanstd(toxin_angles)
                control_mean = np.nanmean(control_angles)
                control_std = np.nanstd(control_angles)
                
                # Calculate angle difference (handle circular data)
                diff = toxin_mean - control_mean
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                
                # Store statistics
                chain_stats[dihedral_type] = {
                    'toxin_mean': toxin_mean,
                    'toxin_std': toxin_std,
                    'control_mean': control_mean,
                    'control_std': control_std,
                    'difference': diff
                }
        
        stats[chain] = chain_stats
    
    # Count flips per residue and system type
    flip_counts = {
        'toxin': {},
        'control': {}
    }
    
    # Process toxin flips
    for system_flips in toxin_flips:
        for chain, chain_flips in system_flips.items():
            if chain not in flip_counts['toxin']:
                flip_counts['toxin'][chain] = {}
            
            for dihedral_name, flips in chain_flips.items():
                if dihedral_name not in flip_counts['toxin'][chain]:
                    flip_counts['toxin'][chain][dihedral_name] = 0
                
                flip_counts['toxin'][chain][dihedral_name] += len(flips)
    
    # Process control flips
    for system_flips in control_flips:
        for chain, chain_flips in system_flips.items():
            if chain not in flip_counts['control']:
                flip_counts['control'][chain] = {}
            
            for dihedral_name, flips in chain_flips.items():
                if dihedral_name not in flip_counts['control'][chain]:
                    flip_counts['control'][chain][dihedral_name] = 0
                
                flip_counts['control'][chain][dihedral_name] += len(flips)
    
    # Create comparison plots
    
    # 1. Mean angle comparison
    plt.figure(figsize=(12, 8))
    
    # Organize data for plotting
    plot_data = []
    
    for chain, chain_stats in stats.items():
        for dihedral_name, dihedral_stats in chain_stats.items():
            # Extract residue position and dihedral type
            parts = dihedral_name.split('_')
            if len(parts) == 2:
                residue_pos, dihedral_type = parts
                
                # Only include carbonyl and psi dihedrals (most relevant for carbonyl orientation)
                if dihedral_type in ['carbonyl', 'psi']:
                    plot_data.append({
                        'Chain': chain,
                        'Residue': residue_pos,
                        'Dihedral': dihedral_type,
                        'Label': f"{chain}:{residue_pos}_{dihedral_type}",
                        'Toxin Mean': dihedral_stats['toxin_mean'],
                        'Toxin Std': dihedral_stats['toxin_std'],
                        'Control Mean': dihedral_stats['control_mean'],
                        'Control Std': dihedral_stats['control_std'],
                        'Difference': dihedral_stats['difference']
                    })
    
    # Sort by absolute difference
    plot_data.sort(key=lambda x: abs(x['Difference']), reverse=True)
    
    # Create DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Save comparison data
    plot_df.to_csv(os.path.join(output_dir, "carbonyl_comparison_data.csv"), index=False)
    
    # Create bar plots
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(plot_df))
        width = 0.35
        
        # Plot toxin means
        toxin_bars = ax.bar(x - width/2, plot_df['Toxin Mean'], width, yerr=plot_df['Toxin Std'],
                          label='Toxin', color='blue', alpha=0.7)
        
        # Plot control means
        control_bars = ax.bar(x + width/2, plot_df['Control Mean'], width, yerr=plot_df['Control Std'],
                            label='Control', color='orange', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Dihedral')
        ax.set_ylabel('Mean Angle (degrees)')
        ax.set_title('Comparison of Carbonyl Orientation Angles')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['Label'], rotation=45, ha='right')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "carbonyl_angle_comparison.png"), dpi=150)
        plt.close()
        
        # Create difference plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot differences
        diff_bars = ax.bar(x, plot_df['Difference'], width=0.6,
                         color=[('red' if d > 0 else 'green') for d in plot_df['Difference']])
        
        # Add labels and title
        ax.set_xlabel('Dihedral')
        ax.set_ylabel('Difference (Toxin - Control) in degrees')
        ax.set_title('Difference in Carbonyl Orientation Angles (Toxin - Control)')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['Label'], rotation=45, ha='right')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Annotate with values
        for i, v in enumerate(plot_df['Difference']):
            ax.text(i, v + (5 if v >= 0 else -5), f"{v:.1f}°", ha='center', va=('bottom' if v >= 0 else 'top'))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "carbonyl_angle_differences.png"), dpi=150)
        plt.close()
    
    # 2. Flip frequency comparison
    if flip_counts['toxin'] or flip_counts['control']:
        # Organize data for plotting
        flip_data = []
        
        # Process each chain
        all_chains = set(list(flip_counts['toxin'].keys()) + list(flip_counts['control'].keys()))
        
        for chain in all_chains:
            # Get all dihedral names for this chain
            toxin_dihedrals = set(flip_counts['toxin'].get(chain, {}).keys())
            control_dihedrals = set(flip_counts['control'].get(chain, {}).keys())
            all_dihedrals = toxin_dihedrals.union(control_dihedrals)
            
            for dihedral_name in all_dihedrals:
                # Get counts
                toxin_count = flip_counts['toxin'].get(chain, {}).get(dihedral_name, 0)
                control_count = flip_counts['control'].get(chain, {}).get(dihedral_name, 0)
                
                # Normalize by number of systems
                toxin_freq = toxin_count / len(toxin_results) if toxin_results else 0
                control_freq = control_count / len(control_results) if control_results else 0
                
                # Calculate difference
                diff = toxin_freq - control_freq
                
                # Extract residue position and dihedral type
                parts = dihedral_name.split('_')
                if len(parts) == 2:
                    residue_pos, dihedral_type = parts
                    
                    flip_data.append({
                        'Chain': chain,
                        'Residue': residue_pos,
                        'Dihedral': dihedral_type,
                        'Label': f"{chain}:{residue_pos}_{dihedral_type}",
                        'Toxin Frequency': toxin_freq,
                        'Control Frequency': control_freq,
                        'Difference': diff
                    })
        
        # Create DataFrame
        flip_df = pd.DataFrame(flip_data)
        
        # Sort by absolute difference
        flip_df = flip_df.sort_values(by='Difference', key=abs, ascending=False)
        
        # Save comparison data
        flip_df.to_csv(os.path.join(output_dir, "carbonyl_flip_comparison.csv"), index=False)
        
        # Create bar plots
        if not flip_df.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(flip_df))
            width = 0.35
            
            # Plot toxin frequencies
            toxin_bars = ax.bar(x - width/2, flip_df['Toxin Frequency'], width,
                              label='Toxin', color='blue', alpha=0.7)
            
            # Plot control frequencies
            control_bars = ax.bar(x + width/2, flip_df['Control Frequency'], width,
                                label='Control', color='orange', alpha=0.7)
            
            # Add labels and title
            ax.set_xlabel('Dihedral')
            ax.set_ylabel('Flip Frequency (per System)')
            ax.set_title('Comparison of Carbonyl Flip Frequencies')
            
            # Set x-ticks
            ax.set_xticks(x)
            ax.set_xticklabels(flip_df['Label'], rotation=45, ha='right')
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "carbonyl_flip_comparison.png"), dpi=150)
            plt.close()
            
            # Create difference plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot differences
            diff_bars = ax.bar(x, flip_df['Difference'], width=0.6,
                             color=[('red' if d > 0 else 'green') for d in flip_df['Difference']])
            
            # Add labels and title
            ax.set_xlabel('Dihedral')
            ax.set_ylabel('Difference (Toxin - Control) in flip frequency')
            ax.set_title('Difference in Carbonyl Flip Frequencies (Toxin - Control)')
            
            # Set x-ticks
            ax.set_xticks(x)
            ax.set_xticklabels(flip_df['Label'], rotation=45, ha='right')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Annotate with values
            for i, v in enumerate(flip_df['Difference']):
                ax.text(i, v + (0.1 if v >= 0 else -0.1), f"{v:.2f}", 
                       ha='center', va=('bottom' if v >= 0 else 'top'))
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "carbonyl_flip_frequency_differences.png"), dpi=150)
            plt.close()
    
    print("Comparison completed and results saved to", output_dir)


def process_directory(base_dir, output_base_dir, system_type="unknown"):
    """
    Process all trajectory files in a directory.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing trajectory files
    output_base_dir : str
        Base directory for output
    system_type : str
        'toxin' or 'control' to categorize the system
    
    Returns:
    --------
    list of tuples
        List of (dihedral_data, carbonyl_flips) for each run
    """
    print(f"Processing {system_type} systems in {base_dir}...")
    
    # Find all trajectory files
    pattern = os.path.join(base_dir, "**", "MD_Aligned.dcd")
    dcd_files = glob.glob(pattern, recursive=True)
    
    if not dcd_files:
        print(f"No DCD files found in {base_dir}")
        return []
    
    print(f"Found {len(dcd_files)} DCD files")
    
    # Process each trajectory file
    results = []
    
    for dcd_file in dcd_files:
        # Determine corresponding PSF file
        dcd_dir = os.path.dirname(dcd_file)
        psf_file = os.path.join(dcd_dir, "step5_input.psf")
        
        if not os.path.exists(psf_file):
            print(f"PSF file not found for {dcd_file}. Skipping.")
            continue
        
        # Determine output directory
        run_name = os.path.basename(dcd_dir)
        output_dir = os.path.join(output_base_dir, f"{system_type}_{run_name}")
        
        # Create analyzer
        try:
            analyzer = CarbonylAnalyzer(psf_file, dcd_file, output_dir, system_type)
            
            # Run analysis
            dihedral_data, carbonyl_flips = analyzer.run_analysis()
            
            # Store results
            results.append((dihedral_data, carbonyl_flips))
            
        except Exception as e:
            print(f"Error processing {dcd_file}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    """Main function to run the carbonyl analysis."""
    parser = argparse.ArgumentParser(description='Analyze carbonyl orientations in K+ channel MD simulations.')
    parser.add_argument('--toxin_dir', default='/home/labs/bmeitan/karbati/rCs1/toxin/toxin',
                       help='Directory containing toxin system data')
    parser.add_argument('--control_dir', default='/home/labs/bmeitan/karbati/Cs1_AF3v/control/control',
                       help='Directory containing control system data')
    parser.add_argument('--output_dir', default='carbonyl_analysis',
                       help='Directory for saving output files')
    parser.add_argument('--threshold', type=float, default=45.0,
                       help='Threshold angle change (degrees) for detecting carbonyl flips')
    parser.add_argument('--single_run', action='store_true',
                       help='Process a single run instead of batch processing')
    parser.add_argument('--dcd_file', default=None,
                       help='Path to DCD file for single run analysis')
    parser.add_argument('--psf_file', default=None,
                       help='Path to PSF file for single run analysis')
    parser.add_argument('--system_type', default='unknown',
                       help='System type (toxin or control) for single run analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.single_run:
        # Process a single run
        if not args.dcd_file or not args.psf_file:
            print("Error: --dcd_file and --psf_file required for single run analysis")
            return
        
        # Create analyzer
        output_dir = os.path.join(args.output_dir, f"{args.system_type}_single_run")
        analyzer = CarbonylAnalyzer(args.psf_file, args.dcd_file, output_dir, args.system_type)
        
        # Run analysis
        analyzer.run_analysis()
        
    else:
        # Process multiple runs
        output_toxin_dir = os.path.join(args.output_dir, 'toxin_runs')
        output_control_dir = os.path.join(args.output_dir, 'control_runs')
        os.makedirs(output_toxin_dir, exist_ok=True)
        os.makedirs(output_control_dir, exist_ok=True)
        
        # Process toxin systems
        toxin_results = process_directory(args.toxin_dir, output_toxin_dir, 'toxin')
        
        # Process control systems
        control_results = process_directory(args.control_dir, output_control_dir, 'control')
        
        # Compare results
        if toxin_results and control_results:
            output_comparison_dir = os.path.join(args.output_dir, 'comparison')
            compare_carbonyl_orientations(toxin_results, control_results, output_comparison_dir)
        else:
            print("Cannot generate comparison - need both toxin and control results")
    
    print("Analysis completed. Results saved to", args.output_dir)


if __name__ == "__main__":
    main()