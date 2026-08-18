
# get input
import sys

#input_file = 'nPC-G_13-19 Dec (1).csv'
input_file = sys.argv[1]
# Check if the file name ends with ".csv"
if input_file.endswith(".csv"):
    # Replace the file name ending with "_calibrated.csv"
    output_file = input_file.replace('.csv', '') + '_calibrated.csv'

else:
    # If the file name doesn't end with ".csv", print an error message and exit
    print("Error: The file name must end with '.csv'")
    sys.exit()


# calibration constants from Slack
# https://c38c.slack.com/archives/C012CR21ZC0/p1674061085419529?thread_ts=1673902789.655769&cid=C012CR21ZC0
m = 0.0000291624603871691
b = 0.0

# open files for reading / writing
with open(input_file, 'r') as f_r:
    with open(output_file, 'w') as f_w:
        # loop through input file
        for n,line in enumerate(f_r):
            if n == 0: # first line / header
                line = line.lower()
                line = line.replace('x,y,z,', 'X.original,Y.original,Z.original,')
                line = line.rstrip('\n')
                line = line.rstrip(',')
                f_w.write(line + '\n') # write header
            else:
                line = line.rstrip('\n')
                line = line.rstrip(',')
                parts = line.split(',') # split by comma 
                
                f_w.write(','.join(parts[0:10]) + ',') # write the parts that don't need to change
                
                parts_float = [float(p) for p in parts[10:]] # convert to numbers rather than strings
		
                
                # perform numerical calibration per wavelength
                calibrated_numbers = [] 
                for p in parts_float:
                    val = p * m + b
                    if val < 0:
                        val = 0.0
                    calibrated_numbers.append(val) # y = mx + b
               
                # convert to strings
                str_write = ''
                for n,p in enumerate(calibrated_numbers):
                    str_write += '%.18f' % p
                    if n < (len(calibrated_numbers) - 1):
                        str_write += ','
                    else:
                        str_write += '\n'
                
                # write string 
                f_w.write(str_write)