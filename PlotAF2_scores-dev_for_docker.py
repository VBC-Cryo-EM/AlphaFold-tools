#!/usr/bin/env python3

import subprocess
import sys
import os
import getpass
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import argparse
import requests
import tqdm
from tabulate import tabulate
import logging


#Directory on /resourced where PDBs are available:
HOST_PDB_PATH = '/Users/matthias.vorlaender/Downloads/PDBs'

# Default root directory for all input data inside the container
DEFAULT_ROOT_DIR = '/usr/src/app/input_data'

# Default output directory is set to '/app', which should be a mounted volume representing the folder where Docker is launched
DEFAULT_OUTPUT_DIR = '/AF2_PPI_tools_output'
PLOTS_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'plots')
HITS_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'hits')
CHIMERAX_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'ChimeraX')
TABLES_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'tables')
LOGS_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'logs')
PDBS_OUTPUT_DIR = os.path.join(CHIMERAX_OUTPUT_DIR, 'PDBs')

# Ensure necessary directories exist
os.makedirs(LOGS_OUTPUT_DIR, exist_ok=True)
os.makedirs(PDBS_OUTPUT_DIR, exist_ok=True)

# Logging setup
def setup_logging(poi_uniprot_id, confidence_threshold):
    log_file = f'{LOGS_OUTPUT_DIR}/script_output_{poi_uniprot_id}_cutoff_{confidence_threshold}.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    return log_file

def log_and_print(message):
    print(message)
    logging.info(message)

# Define all input paths relative to the root directory
DATABASE_PATH = os.path.join(DEFAULT_ROOT_DIR, 'AF_scores_HumanPPI_241003.txt')
PDB_FILE_PATH = os.path.join(DEFAULT_ROOT_DIR, 'precomputed_AF2_predictions/predictions.txt')

# UniProt API URL for querying a UniProt ID
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/{}.txt"
# Function to retrieve UniProt descriptions via the UniProt REST API
def get_uniprot_description(uniprot_id):
    url = UNIPROT_API_URL.format(uniprot_id)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.splitlines()
            for line in lines:
                if line.startswith("DE   RecName: Full="):
                    description = line.split("Full=")[1].rstrip(";")
                    return description
            return "N/A (decrease --labelling threshold to fetch from uniprot)"
        else:
            return f"No results found for {uniprot_id} (Status Code: {response.status_code})"
    except Exception as e:
        return f"Error retrieving description for {uniprot_id}: {str(e)}"

#get predictions from uniprot
def get_uniprot_descriptions_above_threshold(predictions, labelling_threshold):
    descriptions = {}
    ids_for_descriptions = predictions[predictions['Score'] > labelling_threshold]['Other_Protein'].unique()
    log_and_print(f"\nFetching UniProt descriptions for proteins with confidence score > {labelling_threshold}...")
    for uniprot_id in tqdm.tqdm(ids_for_descriptions, desc="..."):
        if uniprot_id not in descriptions:
            descriptions[uniprot_id] = get_uniprot_description(uniprot_id)
    return descriptions

# Optimized Function to mine predictions for the protein of interest (POI)
def get_predictions_for_poi(poi_uniprot_id, data):
    filtered_data = data[(data['Protein1'] == poi_uniprot_id) | (data['Protein2'] == poi_uniprot_id)].copy()
    if filtered_data.empty:
        log_and_print(f"Error: The protein {poi_uniprot_id} does not exist in the database.")
        sys.exit(1)
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        fig.write_html(output_file)
        log_and_print(f"Plot saved to .{output_file}")
    except Exception as e:
        log_and_print(f"Error saving plot to {output_file}: {e}")
        sys.exit(1)

def write_poi_to_file(poi_uniprot_id, POI_name, output_dir):
    file_name = f"{poi_uniprot_id}_{POI_name}.txt"
    poi_output_file_path = os.path.join(output_dir, "hits", file_name)
    os.makedirs(os.path.dirname(poi_output_file_path), exist_ok=True)
    
    with open(poi_output_file_path, 'w') as f:
        f.write(poi_uniprot_id)
    #log_and_print(f"POI text file saved to .{poi_output_file_path}")
    return poi_output_file_path

def chimeraX_script(poi_uniprot_id, output_dir, available_pdbs, POI_name):

    chimerax_script = [f"#open Alphafold model for the POI using its uniprot ID",
                       f"alphafold fetch {poi_uniprot_id}",
                       f"cd PDBs"]

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
    log_and_print(f"A ChimeraX script for visualising all available pre-computed structures was saved to .{chimerax_output_path}")

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
    #log_and_print(f"Text file with potential interactors above cutoff saved to .{output_file_path}")

    # Return the output file path
    return output_file_path

# Function to write interacting proteins to an HTML file
def write_hits_to_html(predictions, output_dir, poi_uniprot_id, POI_name, confidence_threshold):
    filtered_predictions = predictions[predictions['Score'] > confidence_threshold].copy()
    table_data = []
    for _, row in filtered_predictions.iterrows():
        description = row['Description'] if row['Description'] != "Description not found" else "N/A"
        table_data.append([row['Other_Protein'], description, row['Score'], row['PDB_File']])
    headers = ["Interacting Protein", "UniProt Description", "Confidence Score", "PDB Filename"]
    html_table = tabulate(table_data, headers=headers, tablefmt="html")
    file_name = f"{poi_uniprot_id}_{POI_name}_hits_above_{confidence_threshold}.html"
    output_file_path = os.path.join(TABLES_OUTPUT_DIR, file_name)
    with open(output_file_path, 'w') as f:
        f.write("<html><body>")
        f.write("<h2>Putative Interacting Proteins Above Confidence Threshold</h2>")
        f.write(html_table)
        f.write("</body></html>")
    log_and_print(f"HTML table with hits above cutoff saved to .{output_file_path}")

# Function to read PDB availability from a text file
def read_pdb_availability(pdb_file_path):
    pdb_files = []
    with open(pdb_file_path, 'r') as f:
        pdb_lines = f.readlines()
    pdb_uniprot_pairs = set()
    for line in pdb_lines:
        parts = line.strip().split('__')
        if len(parts) == 2:
            uniprot1 = parts[0].split('_')[0]
            uniprot2 = parts[1].split('_')[0]
            pdb_uniprot_pairs.add((uniprot1, uniprot2))
            pdb_uniprot_pairs.add((uniprot2, uniprot1))
            pdb_files.append((uniprot1, uniprot2, line.strip()))
    log_and_print(f"Predictions with available PDBs loaded")
    return pdb_uniprot_pairs, pdb_files

# Function to generate copy commands for PDB files and save them to a script
def generate_copy_commands(available_pdbs, output_file_path):
    available_pdb_filenames = available_pdbs['PDB_File'].unique()
    with open(output_file_path, 'w') as f:
        for pdb_filename in available_pdb_filenames:
            source_path = os.path.join(HOST_PDB_PATH, pdb_filename)
            dest_path = os.path.join(PDBS_OUTPUT_DIR, pdb_filename)
            # Check if the file doesn't already exist before copying
            command = f"[ ! -f '{dest_path}' ] && cp -v '{source_path}' '.{dest_path}'\n"
            f.write(command)
    #log_and_print(f"Generated copy commands script at .{output_file_path}")

# Optimized plotting function with UniProt descriptions for proteins above a set threshold
def plot_poi_predictions_with_density_jitter(poi_uniprot_id, POI_name, data, file_path, output_dir, pdb_file_path, labelling_threshold, width, height):
    predictions = get_predictions_for_poi(poi_uniprot_id, data)
    log_and_print(f"Number of predictions for {poi_uniprot_id} in the database: {len(predictions)}")
    predictions = apply_density_based_jitter(predictions, 'Score')
    
    # Read PDB availability
    pdb_uniprot_pairs, pdb_files = read_pdb_availability(pdb_file_path)

    
    # Get descriptions above the specified threshold
    descriptions = get_uniprot_descriptions_above_threshold(predictions, labelling_threshold)
    
    # Map descriptions back to the DataFrame (show "Description not found" for others)
    predictions['Description'] = predictions['Other_Protein'].map(lambda x: descriptions.get(x, "Description not found"))
    
    # Add PDB availability information with improved matching logic
    def check_pdb_availability(row, poi_id, pdb_pairs, pdb_files):
        other_protein_id = row['Other_Protein'].split('_')[0]
        for pdb_entry in pdb_files:
            if (other_protein_id, poi_id) in pdb_pairs or (poi_id, other_protein_id) in pdb_pairs:
                if (pdb_entry[0], pdb_entry[1]) == (other_protein_id, poi_id) or (pdb_entry[1], pdb_entry[0]) == (poi_id, other_protein_id):
                    return True, pdb_entry[2]
        return False, "N/A"
    
    predictions['PDB_Available'], predictions['PDB_File'] = zip(*predictions.apply(lambda row: check_pdb_availability(row, poi_uniprot_id, pdb_uniprot_pairs, pdb_files), axis=1))
    
    # Count how many predictions are listed in the PDB availability file
    pdb_available_count = predictions['PDB_Available'].sum()
    log_and_print(f"Number of predictions for {poi_uniprot_id} with available PDB files: {pdb_available_count}")
  
    # Plot all predictions with a single color scale
    fig = px.scatter(predictions, x="x_jitter", y="Score",
                     color="Score", color_continuous_scale="Viridis",
                     hover_data={
                         "Other_Protein": True,
                         "Description": True,
                         "Score": True,
                         "PDB_Available": True,
                         "PDB_File": True,
                         "x_jitter": False
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
    # Update layout to include a legend entry for PDB availability
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=6, line=dict(width=1, color='black'), color='white'),
            name='Pre-computed PDB available for download'
        )
    )

    # Update layout to include a legend with white background
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, automargin=True, title=""),
        legend=dict(title='Legend', itemsizing='constant', title_font=dict(size=14), font=dict(size=12), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="white"),
        yaxis=dict(automargin=True, title="Confidence Score"),
        showlegend=True,
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    db_name = os.path.basename(file_path).split('.')[0]
    file_name = f"{db_name}_{poi_uniprot_id}_{POI_name}_predictions.html"
    save_plot_as_html(fig, PLOTS_OUTPUT_DIR, file_name)

    # Save the table of predictions as an HTML file
    write_hits_to_html(predictions, TABLES_OUTPUT_DIR, poi_uniprot_id, POI_name, labelling_threshold)

    # Print a message indicating which predictions above the cutoff have pre-computed results available
    available_pdbs = predictions[predictions['PDB_Available'] & (predictions['Score'] > labelling_threshold)]
    
    if not available_pdbs.empty:
        log_and_print("The following interacting proteins have pre-computed PDB results available:")
        table_data = [[row['Other_Protein'], row['Score'], row['PDB_File']] for _, row in available_pdbs.iterrows()]
        headers = ["Interacting Protein", "Score", "PDB Filename"]
        log_and_print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        log_and_print("No pre-computed PDB results are available for predictions above the cutoff.")
    
    return available_pdbs

#Generate command for Dominiks AF screening
def generate_shell_command(poi_file_path, output_file_path, poi_uniprot_id, POI_name, confidence_threshold):
    command = (f"/resources/colabfold/software/ht-colabfold/alphafold_batch.sh -1 .{poi_file_path} -2 .{output_file_path} -O $PWD{DEFAULT_OUTPUT_DIR}/screens/{poi_uniprot_id}_{POI_name}_vs_hits_above_{confidence_threshold}")
    
    # Print the command (but do not execute it)
    log_and_print("\nNote that the plot shows available confidence scores, but most will lack an associated PDB. To re-compute the structures of all hits above your cutoff using Dominik's script, execute the following command:")
    log_and_print("\n ###########BEGIN COMMAND###########")   
    log_and_print("\n")
    log_and_print(command)
    log_and_print("\n ###########END COMMAND###########")
    log_and_print("\n")

# Main execution logic
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot protein-protein interactions and output results.')
    parser.add_argument('--poi', type=str, required=True, help='UniProt ID of the protein of interest (POI)')
    parser.add_argument('--cutoff', type=float, default=0.5, help='Confidence score cutoff (default: 0.5)')
    parser.add_argument('--POI_name', type=str, required=True, help='Custom name to append to the UniProt ID')
    parser.add_argument('--chimeraX', action='store_true', help='Generate ChimeraX script for visualizing pre-computed predictions.')
    parser.add_argument('--labelling_threshold', type=float, default=0.3, help='Threshold above which UniProt descriptions are fetched (default: 0.3).')
    parser.add_argument('--plot_width', type=int, default=1200, help='Width of the plot (default: 1200)')
    parser.add_argument('--plot_height', type=int, default=800, help='Height of the plot (default: 800)')

    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging(args.poi, args.cutoff)
    log_and_print("\n")
    log_and_print(f"Logging output to {log_file}")

    # Update paths based on the root directory
    database = os.path.join(DEFAULT_ROOT_DIR, 'AF_scores_HumanPPI_241003.txt')
    pdb_file = os.path.join(DEFAULT_ROOT_DIR, 'precomputed_AF2_predictions/predictions.txt')

    # Load data once
    data = pd.read_csv(database, sep="\t", header=None, names=["Protein_Pair", "Score"])
    data[['Protein1', 'Protein2']] = data['Protein_Pair'].str.split('_', expand=True)

    # Run the plot function
    log_and_print("\n")
    available_pdbs = plot_poi_predictions_with_density_jitter(args.poi, args.POI_name, data, database, DEFAULT_OUTPUT_DIR, pdb_file, labelling_threshold=args.labelling_threshold, width=args.plot_width, height=args.plot_height)

    # Generate copy commands for available PDB files
    # Generate a shell script to copy PDB files for available PDBs
    copy_commands_file_path = os.path.join(DEFAULT_OUTPUT_DIR, "copy_pdb_commands.sh")
    generate_copy_commands(available_pdbs, copy_commands_file_path)

    # Provide instructions for running the generated shell script
    #log_and_print(f"To copy the required PDB files, please run the following command on the host:\n")
    #log_and_print(f"sh .{copy_commands_file_path}\n")

    # Write the text file with interacting proteins above the confidence threshold
    output_file_path = write_interacting_proteins_to_file(args.poi, args.POI_name, data, database, args.cutoff, DEFAULT_OUTPUT_DIR)

    # Write the POI to a text file
    poi_file_path = write_poi_to_file(args.poi, args.POI_name, DEFAULT_OUTPUT_DIR)
    
    # Generate and log_and_print the shell command
    generate_shell_command(poi_file_path, output_file_path, args.poi, args.POI_name, args.cutoff)


    # Generate ChimeraX script if requested
    if args.chimeraX:
        chimeraX_script(args.poi, CHIMERAX_OUTPUT_DIR, available_pdbs, args.POI_name)


