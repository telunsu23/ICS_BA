import pandas as pd


def move_columns_to_end(file_path, columns_to_move, output_path=None):
    """
    Moves specified columns to the end of a DataFrame and saves the result to a new CSV file.

    Args:
        file_path (str): The path to the source CSV file.
        columns_to_move (list): A list of column names to move to the end.
        output_path (str, optional): The path to save the new CSV file.
                                     If None, it overwrites the original file.
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Check if all columns to move exist
        missing_columns = [col for col in columns_to_move if col not in df.columns]
        if missing_columns:
            print(f"Warning: The following columns were not found and will be ignored: {missing_columns}")
            # Filter out columns that do not exist
            columns_to_move = [col for col in columns_to_move if col in df.columns]

        # Separate columns to move and remaining columns
        cols_to_keep = [col for col in df.columns if col not in columns_to_move]

        # Reorder DataFrame, placing columns to move at the end
        df_reordered = df[cols_to_keep + columns_to_move]

        # Determine output path
        if output_path is None:
            output_path = file_path

        # Save the new CSV file
        df_reordered.to_csv(output_path, index=False)
        print(f"File processing complete! Specified columns moved to the end and saved to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check if the file path is correct.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


# Path to the CSV file to be processed
csv_file = 'train.csv'

columns_to_move = [
    'P1_PP01AD', 'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR',
    'P1_PP02D', 'P1_PP02R', 'P1_STSP', 'P2_ATSW_Lamp',
    'P2_AutoGO', 'P2_Emerg', 'P2_MASW', 'P2_MASW_Lamp',
    'P2_ManualGO', 'P2_OnOff', 'P2_TripEx', 'Attack'
]

# Call the function
move_columns_to_end(file_path=csv_file, columns_to_move=columns_to_move)

# P1_PP01AD, P1_PP01AR, P1_PP01BD, P1_PP01BR, P1_PP02D, P1_PP02R, P1_SOL01D, P1_SOL03D, P1_STSP, P2_ATSW_Lamp, P2_AutoGO
# P2_Emerg, P2_MASW, P2_MASW_Lamp, P2_ManualGO, P2_OnOff, P2_TripEx: 17 actuators, 69 sensors