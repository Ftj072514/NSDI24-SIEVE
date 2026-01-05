import struct
import sys

def parse_trace(filename, num_records=20):
    # struct format from oracleGeneralBin.h:
    # uint32_t real_time;       (4 bytes)
    # uint64_t obj_id;          (8 bytes)
    # uint32_t obj_size;        (4 bytes)
    # int64_t next_access_vtime; (8 bytes)
    # Total size: 24 bytes
    record_struct = struct.Struct('<IQIq')
    record_size = record_struct.size

    print(f"{'Time':<10} {'Obj ID':<20} {'Size':<10} {'Next Access':<20}")
    print("-" * 60)

    try:
        with open(filename, 'rb') as f:
            count = 0
            while count < num_records:
                chunk = f.read(record_size)
                if len(chunk) < record_size:
                    break
                
                real_time, obj_id, obj_size, next_vtime = record_struct.unpack(chunk)
                
                # Format next_access_vtime for better readability if check for MAX_INT
                next_vtime_str = str(next_vtime)
                if next_vtime == 9223372036854775807: # INT64_MAX
                    next_vtime_str = "INF"

                print(f"{real_time:<10} {obj_id:<20} {obj_size:<10} {next_vtime_str:<20}")
                count += 1
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error parsing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 parse_trace.py <trace_file>")
        sys.exit(1)
    
    parse_trace(sys.argv[1])
