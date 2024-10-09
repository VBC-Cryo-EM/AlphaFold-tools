#!/usr/bin/env python3
#works but no legend

import subprocess
import sys
import os
import getpass
import pandas as pd
import plotly.express as px
import numpy as np
import argparse
import requests

# Default root directory for all input paths
DEFAULT_ROOT_DIR = '/groups/plaschka/shared/alphafold/HumanPPI_database'
VOLUMES_PREFIX = '/Volumes/'

# Define all input paths relative to the root directory
DATABASE_PATH = os.path.join(DEFAULT_ROOT_DIR, 'AF_scores_HumanPPI_241003.txt')
PDB_FILE_PATH = os.path.join(DEFAULT_ROOT_DIR, 'precomputed_AF2_predictions/predictions.txt')
PRECOMPUTED_DIR = os.path.join(DEFAULT_ROOT_DIR, 'precomputed_AF2_predictions/PDBs')

# Default output directory is the user's current working directory
DEFAULT_OUTPUT_DIR = os.getcwd()
PLOTS_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'plots')
HITS_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'hits')
CHIMERAX_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'ChimeraX')

# UniProt API URL for querying a UniProt ID
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/{}.txt"

# Function to retrieve UniProt descriptions via the UniProt REST API
def get_uniprot_description(uniprot_id):
    url = UNIPROT_API_URL.format(uniprot_id)
    try:
        # Make a GET request to the UniProt API
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the response to find the "DE" (Description) field
            lines = response.text.splitlines()
            for line in lines:
                if line.startswith("DE   RecName: Full="):
                    # Extract the protein description
                    description = line.split("Full=")[1].rstrip(";")
                    return description
            return "Description not found"
        else:
            return f"No results found for {uniprot_id} (Status Code: {response.status_code})"
    except Exception as e:
        return f"Error retrieving description for {uniprot_id}: {str(e)}"

# Optimized Function to mine predictions for the protein of interest (POI)
def get_predictions_for_poi(poi_uniprot_id, data):
    # Filter rows where the POI appears in either Protein1 or Protein2
    filtered_data = data[(data['Protein1'] == poi_uniprot_id) | (data['Protein2'] == poi_uniprot_id)].copy()
    
    if filtered_data.empty:
        print(f"Error: The protein {poi_uniprot_id} does not exist in the database.")
        sys.exit(1)
    
    # Set the 'Other_Protein' column using vectorized operations
    filtered_data['Other_Protein'] = np.where(filtered_data['Protein1'] == poi_uniprot_id, filtered_data['Protein2'], filtered_data['Protein1'])
    
    return filtered_data

# Function to apply jitter based on point density
def apply_density_based_jitter(sorted_data, y_col, base_jitter=0.005, max_jitter=0.05):
    y_rounded = sorted_data[y_col].round(1)
    counts = y_rounded.value_counts()
    sorted_data['point_density'] = y_rounded.map(counts)
    sorted_data['jitter_amount'] = base_jitter + ((sorted_data['point_density'] - 1) / (counts.max() - 1)) * (max_jitter - base_jitter)
    sorted_data['x_jitter'] = np.random.uniform(-1, 1, size=len(sorted_data)) * sorted_data['jitter_amount']
    return sorted_data

# Function to save plot as an HTML file
def save_plot_as_html(fig, output_dir, file_name):
    output_file = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")

# Function to read PDB availability from a text file
def read_pdb_availability(pdb_file_path):
    pdb_files = []
    with open(pdb_file_path, 'r') as f:
        pdb_lines = f.readlines()
    pdb_uniprot_pairs = set()
    for line in pdb_lines:
        parts = line.strip().split('__')
        if len(parts) == 2:
            uniprot1 = parts[0].split('_')[0]  # Extract the UniProt ID before the first underscore
            uniprot2 = parts[1].split('_')[0]  # Extract the UniProt ID before the first underscore
            pdb_uniprot_pairs.add((uniprot1, uniprot2))
            pdb_uniprot_pairs.add((uniprot2, uniprot1))
            pdb_files.append((uniprot1, uniprot2, line.strip()))
    print(f"Predictions with available PDBs loaded")  # Debugging message
    return pdb_uniprot_pairs, pdb_files

# Function to write interacting proteins to a text file and return the file path
def write_interacting_proteins_to_file(poi_uniprot_id, POI_name, data, file_path, confidence_threshold, output_dir):
    # Get predictions for the protein of interest
    predictions = get_predictions_for_poi(poi_uniprot_id, data)
    # Filter predictions above the confidence threshold
    filtered_predictions = predictions[predictions['Score'] > confidence_threshold]
    # Get the unique interacting proteins
    interacting_proteins = filtered_predictions['Other_Protein'].unique()
    # Extract the database name from the file path
    db_name = os.path.basename(file_path).split('.')[0]
    # Create the file name for the text file
    file_name = f"{db_name}_{poi_uniprot_id}_{POI_name}_interacting_proteins_above_{confidence_threshold}.txt"
    # Create the output file path, including the "hits" folder
    output_file_path = os.path.join(output_dir, "hits", file_name)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Write the interacting proteins to the file
    with open(output_file_path, 'w') as f:
        f.write("\n".join(interacting_proteins))
    # Print confirmation message
    print(f"Text file with potential interactors above cutoff saved to {output_file_path}")
    
    # Return the output file path
    return output_file_path


# Optimized plotting function with UniProt descriptions for proteins above a set threshold
def plot_poi_predictions_with_density_jitter(poi_uniprot_id, POI_name, data, file_path, output_dir, pdb_file_path, score_for_descriptions=0.3, width=1200, height=800):
    # No filtering based on user threshold here; plot all points
    predictions = get_predictions_for_poi(poi_uniprot_id, data)
    print(f"Number of predictions for {poi_uniprot_id} in the database: {len(predictions)}")  # Debugging message
    predictions = apply_density_based_jitter(predictions, 'Score')
    
    # Read PDB availability
    pdb_uniprot_pairs, pdb_files = read_pdb_availability(pdb_file_path)
    
    # Filter for retrieving descriptions (confidence score above 0.3)
    ids_for_descriptions = predictions[predictions['Score'] > score_for_descriptions]['Other_Protein'].unique()
    
    descriptions = {}
    print(f"\nFetching UniProt descriptions for proteins with confidence score > {score_for_descriptions}...")
    
    for uniprot_id in ids_for_descriptions:
        if uniprot_id not in descriptions:
            descriptions[uniprot_id] = get_uniprot_description(uniprot_id)
    
    # Map descriptions back to the DataFrame (show "Description not found" for others)
    predictions['Description'] = predictions['Other_Protein'].map(lambda x: descriptions.get(x, "Description not found"))
    
    # Add PDB availability information with improved matching logic
    def check_pdb_availability(row, poi_id, pdb_pairs, pdb_files):
        other_protein_id = row['Other_Protein'].split('_')[0]  # Ensure we are using the UniProt ID without suffix
        for pdb_entry in pdb_files:
            if (other_protein_id, poi_id) in pdb_pairs or (poi_id, other_protein_id) in pdb_pairs:
                if (pdb_entry[0], pdb_entry[1]) == (other_protein_id, poi_id) or (pdb_entry[1], pdb_entry[0]) == (poi_id, other_protein_id):
                    return True, pdb_entry[2]
        return False, None
    
    predictions['PDB_Available'], predictions['PDB_File'] = zip(*predictions.apply(lambda row: check_pdb_availability(row, poi_uniprot_id, pdb_uniprot_pairs, pdb_files), axis=1))
    
    # Count how many predictions are listed in the PDB availability file
    pdb_available_count = predictions['PDB_Available'].sum()
    print(f"Number of predictions for {poi_uniprot_id} with available PDB files: {pdb_available_count}")  # Debugging message
  
    # Plot all predictions with a single color scale
    fig = px.scatter(predictions, x="x_jitter", y="Score",
                     color="Score", color_continuous_scale="Viridis",
                     hover_data={
                         "Other_Protein": True,
                         "Description": True,
                         "Score": True,
                         "PDB_Available": True,
                         "PDB_File": True,
                         "x_jitter": False  # Exclude jitter from hover info
                     },
                     labels={
                         "Score": "Confidence Score",
                         "Other_Protein": "Interacting Protein",
                         "Description": "Protein Description",
                         "PDB_Available": "PDB Available",
                         "PDB_File": "PDB Filename"
                     },
                     title=f'Protein-Protein Interaction Predictions for {poi_uniprot_id}',
                     width=width,
                     height=height)

    # Update marker style for points with PDB available
    fig.update_traces(marker=dict(symbol='circle', size=6,
                                  line=dict(width=np.where(predictions['PDB_Available'], 2, 0), color='black')),
                      selector=dict(mode='markers'))

    # Update layout to include a legend
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, automargin=True, title=""),
        legend=dict(title='Legend', itemsizing='constant', title_font=dict(size=14), font=dict(size=12), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.2),
        yaxis=dict(automargin=True, title="Confidence Score"),
        showlegend=True,
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    db_name = os.path.basename(file_path).split('.')[0]
    file_name = f"{db_name}_{poi_uniprot_id}_{POI_name}_predictions.html"
    #fig.show()
    save_plot_as_html(fig, output_dir, file_name)

    # Print a message indicating which predictions above the cutoff have pre-computed results available
    precomputed_dir = pdb_file_path.replace('/groups/', '/Volumes/')
    available_pdbs = predictions[predictions['PDB_Available'] & (predictions['Score'] > score_for_descriptions)]
    print(f"\nPre-computed results for predictions above the cutoff are available at: {precomputed_dir}")
    if not available_pdbs.empty:
        print("The following interacting proteins have pre-computed PDB results available:")
        for _, row in available_pdbs.iterrows():
            print(f"- {row['Other_Protein']} (Score: {row['Score']}), PDB File: {row['PDB_File']}")
    else:
        print("No pre-computed PDB results are available for predictions above the cutoff.")

    return available_pdbs

# Function to write the POI into a text file and return the file path
def write_poi_to_file(poi_uniprot_id, POI_name, output_dir):
    file_name = f"{poi_uniprot_id}_{POI_name}.txt"
    poi_output_file_path = os.path.join(output_dir,  file_name)
    os.makedirs(os.path.dirname(poi_output_file_path), exist_ok=True)
    
    with open(poi_output_file_path, 'w') as f:
        f.write(poi_uniprot_id)
    print(f"POI text file saved to {poi_output_file_path}")
    return poi_output_file_path

# Function to generate and print the shell command (without executing it)
def generate_shell_command(poi_file_path, output_file_path, poi_uniprot_id, POI_name, confidence_threshold):
    username = getpass.getuser()  # Get the current username
    command = f"/resources/colabfold/software/ht-colabfold/alphafold_batch.sh \n" \
              f"-1 {poi_file_path} \n" \
              f"-2 {output_file_path} \n" \
              f"-O /groups/plaschka/shared/alphafold/{username}/{POI_name}_{poi_uniprot_id}_vs_hits_above_{confidence_threshold}"
    
    # Print the command (but do not execute it)
    print("\nNote that these are only the pre-computed confidence scores. To re-compute the structures using Dominik's script, execute the following command:")
    print(command)

# Function to generate the ChimeraX work directory based on user input
def get_chimerax_work_dir(precomputed_dir, local_path=None):
    if local_path:
        return local_path
    else:
        return precomputed_dir.replace('/groups/', VOLUMES_PREFIX)

# Function to generate ChimeraX script to load and align available PDBs
def chimeraX_script(poi_uniprot_id, precomputed_dir, output_dir, available_pdbs, POI_name, local_PDB_path=None):
    chimerax_work_dir = get_chimerax_work_dir(precomputed_dir, local_PDB_path)

    chimerax_script = [f"#open Alphafold model for the POI using its uniprot ID",
                       f"alphafold fetch {poi_uniprot_id}",
                       f"cd {chimerax_work_dir}"]

    # Add commands to open available PDBs
    model_index = 2
    for pdb_file in available_pdbs['PDB_File']:
        chimerax_script.append(f"open {pdb_file}")
        model_index += 1

    # Add commands to align PDBs to the reference Alphafold model
    for i in range(2, model_index):
        chimerax_script.append(f"mm #{i} to #1")

    # Add ChimeraX settings
    chimerax_script.extend([
        "view orient",
        "hide #1 models",
        "hide atoms",
        "show cartoon",
        "lighting soft",
        "set bgColor white",
        "graphics silhouettes true"
    ])

    # Add color settings for POI and interactors
    model_index = 2
    for i, row in available_pdbs.iterrows():
        pdb_file = row['PDB_File']
        if poi_uniprot_id in pdb_file.split('__')[0]:
            chimerax_script.append(f"color #{model_index}/A rebecca purple")
            chimerax_script.append(f"color #{model_index}/B yellow")
        else:
            chimerax_script.append(f"color #{model_index}/B rebecca purple")
            chimerax_script.append(f"color #{model_index}/A yellow")
        model_index += 1

    # Write the ChimeraX script to a file
    chimerax_file_name = f"{poi_uniprot_id}_{POI_name}_chimerax_script.cxc"
    chimerax_output_path = os.path.join(output_dir, chimerax_file_name)
    os.makedirs(os.path.dirname(chimerax_output_path), exist_ok=True)
    with open(chimerax_output_path, 'w') as f:
        f.write("\n".join(chimerax_script))
    print(f"ChimeraX script saved to {chimerax_output_path}")

# Main execution logic
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot AlphaFold2 confidence scores from http://prodata.swmed.edu/humanPPI/download for your protein of interest (POI).')
    parser.add_argument('--poi', type=str, required=True, help='UniProt ID of  your POI')
    parser.add_argument('--cutoff', type=float, default=0.5, help='Confidence score cutoff above which hits are written out to a text file (default: 0.5)')
    parser.add_argument('--POI_name', type=str, required=True, help='Protein name of your POI. Will be appended to the UniProt ID in all output files')
    parser.add_argument('--root_dir', type=str, default=DEFAULT_ROOT_DIR, help=f'Root directory for all input data (default: {DEFAULT_ROOT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help=f'Output directory for all generated files (default: Your current working directory)')
    parser.add_argument('--chimeraX', action='store_true', help='Generate ChimeraX script for visualizing all available pre-computed predictions')
    parser.add_argument('--local_PDB_path', type=str, help='Can be used to set to correctly set PDB file paths in ChimeraX scripts. This needs to match the file paths to pre-computed PDBs as seen on your local machine')

    args = parser.parse_args()

    # Update paths based on the root directory and output directory
    database = os.path.join(args.root_dir, 'AF_scores_HumanPPI_241003.txt')
    pdb_file = os.path.join(args.root_dir, 'precomputed_AF2_predictions/predictions.txt')
    precomputed_dir = os.path.join(args.root_dir, 'precomputed_AF2_predictions/PDBs')
    plots_output_dir = os.path.join(args.output_dir, 'plots')
    hits_output_dir = os.path.join(args.output_dir, 'hits')
    chimerax_output_dir = os.path.join(args.output_dir, 'ChimeraX')

    # Load data once
    data = pd.read_csv(database, sep="\t", header=None, names=["Protein_Pair", "Score"])
    data[['Protein1', 'Protein2']] = data['Protein_Pair'].str.split('_', expand=True)

    # Run the plot function
    available_pdbs = plot_poi_predictions_with_density_jitter(args.poi, args.POI_name, data, database, plots_output_dir, pdb_file)

    # Write the text file with interacting proteins above the confidence threshold
    output_file_path = write_interacting_proteins_to_file(args.poi, args.POI_name, data, database, args.cutoff, hits_output_dir)

    # Write the POI to a text file
    poi_file_path = write_poi_to_file(args.poi, args.POI_name, hits_output_dir)

    # Generate and print the shell command
    generate_shell_command(poi_file_path, output_file_path, args.poi, args.POI_name, args.cutoff)

    # If requested, generate ChimeraX script
    if args.chimeraX:
        chimeraX_script(args.poi, precomputed_dir, chimerax_output_dir, available_pdbs, args.POI_name, args.local_PDB_path)
