{description}

Raw dataset path: {input_path}
Clean output path: {output_path}

Cleaning rules:

1. Keep only rows with status == "active".
2. Drop rows where score is empty.
3. Preserve the header and the original column order.

Use read_dataset_file to load the CSV, clean it with python_expression, persist the cleaned CSV with write_clean_file, then submit a JSON object containing rows_kept and average_score via submit_answer.
