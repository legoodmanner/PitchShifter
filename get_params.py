# t0 : 1, t1 : 2, p_mult : 1.5
# t0 : 3, t1 : 4, p_mult : 0.5

# -> [(1, 2, 1.5), (3, 4, 0.5)]
from pdb import set_trace as bp



def get_params(file_path):
    # Initialize an empty list to hold the tuples
    data_tuples = []
    
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line into components based on ': ' and ', '
            parts = line.replace('\n', '').split(', ')
            # Extract t0, t1, and p_mult values from the parts
            t0 = float(parts[0].split(': ')[1])
            t1 = float(parts[1].split(': ')[1])
            p_mult = float(parts[2].split(': ')[1])
            # Append a tuple of (t0, t1, p_mult) to the list
            data_tuples.append((t0, t1, p_mult))
    
    # Return the list of tuples
    return data_tuples

# Replace 'path/to/your/file.txt' with the actual file path
# Call the function and print the result
# result = get_params()
# print(result)
