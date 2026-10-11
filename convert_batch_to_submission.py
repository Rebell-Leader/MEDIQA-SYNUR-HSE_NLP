import json
import sys
import os

def convert(input_path, output_path):
    print(f"Converting {input_path} -> {output_path}")
    count = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if 'custom_id' in data and 'response' in data:
                    p_id = data['custom_id']

                    # Check for null response (failed request)
                    if data.get('response') is None:
                        print(f"Skipping failed item {p_id}")
                        continue

                    content_str = data['response']['body']['choices'][0]['message']['content']

                    # Cleanup markdown
                    if content_str.startswith('```json'):
                        content_str = content_str.replace('```json', '', 1)
                        if content_str.endswith('```'):
                            content_str = content_str[:-3]
                    elif content_str.startswith('```'):
                        content_str = content_str.replace('```', '', 1)
                        if content_str.endswith('```'):
                            content_str = content_str[:-3]

                    content = json.loads(content_str.strip())

                    out_item = {
                        'id': p_id,
                        'observations': content.get('observations', [])
                    }
                    fout.write(json.dumps(out_item) + '\n')
                    count += 1
                else:
                    # Pass through if it looks like it's already in the right format?
                    # Or warn. For now let's strict check batch format.
                    print(f"Skipping non-batch line")
            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"Converted {count} items.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_batch_to_submission.py input_file output_file")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])
