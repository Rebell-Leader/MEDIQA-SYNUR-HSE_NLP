import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from openai import OpenAI

STATUS_FILE = "outputs/status_check.json"

class BatchManager:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.status_file = os.path.join(output_dir, "status_check.json")
        os.makedirs(output_dir, exist_ok=True)
        self._load_status()

    def _load_status(self):
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    self.tracker = json.load(f)
            except json.JSONDecodeError:
                self.tracker = {}
        else:
            self.tracker = {}

    def _save_status(self):
        with open(self.status_file, 'w') as f:
            json.dump(self.tracker, f, indent=4)

    def submit_batch(self, client: OpenAI, file_path: str, model_key: str, experiment_name: str) -> str:
        """
        Uploads and submits a batch, tracking it in status_check.json.
        """
        filename = os.path.basename(file_path)

        # Check if already tracked/submitted
        for bid, info in self.tracker.items():
            if info['file'] == filename and info['model'] == model_key and info['experiment'] == experiment_name:
                print(f"Batch for {filename} ({experiment_name}) already tracked as {bid}. Status: {info['status']}")
                return bid

        print(f"Uploading {filename}...")
        with open(file_path, "rb") as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )

        print(f"Creating batch job for {experiment_name}...")
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "experiment": experiment_name,
                "model": model_key
            }
        )

        self.tracker[batch_job.id] = {
            "model": model_key,
            "experiment": experiment_name,
            "file": filename,
            "file_id": batch_input_file.id,
            "status": "submitted", # initial status
            "created_at": datetime.now().isoformat(),
            "output_file_id": None
        }
        self._save_status()
        print(f"Submitted batch {batch_job.id}")
        return batch_job.id

    def update_statuses(self, client: OpenAI):
        """
        Checks status of all non-terminal batches and downloads results if ready.
        Does NOT overwrite existing result files if they exist (unless partial).
        """
        updated = False
        print("Checking batch statuses...")

        for batch_id, info in self.tracker.items():
            if info['status'] in ['completed', 'failed', 'expired', 'cancelled']:
                continue

            try:
                batch = client.batches.retrieve(batch_id)
                current_status = batch.status
                info['status'] = current_status
                updated = True

                print(f"Batch {batch_id} ({info['experiment']}): {current_status}")

                if current_status == 'completed':
                    info['output_file_id'] = batch.output_file_id
                    self._download_results(client, batch_id, info)

                elif current_status in ['failed', 'expired', 'cancelled']:
                     print(f"Batch {batch_id} failed/cancelled.")

            except Exception as e:
                print(f"Error checking batch {batch_id}: {e}")

        if updated:
            self._save_status()

    def _download_results(self, client: OpenAI, batch_id: str, info: Dict):
        output_file_id = info.get('output_file_id')
        if not output_file_id:
            return

        # Construct distinct output filename
        # e.g. outputs/results_gpt-5-nano_0-shot_valid.jsonl
        out_filename = f"results_{info['model']}_{info['experiment']}.jsonl"
        out_path = os.path.join(self.output_dir, out_filename)

        if os.path.exists(out_path):
            print(f"Results file {out_filename} already exists. Skipping download to prevent overwrite.")
            return

        print(f"Downloading results for {batch_id} to {out_path}...")
        try:
            content = client.files.content(output_file_id).content
            with open(out_path, 'wb') as f:
                f.write(content)
            print(f"Downloaded {len(content)} bytes.")
        except Exception as e:
            print(f"Failed to download results: {e}")

