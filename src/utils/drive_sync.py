"""
src/utils/drive_sync.py

Best-effort upload of local checkpoint files to a Google Drive folder via a
service account -- the write-side counterpart to generate_dashboard.py's
gdown-based read-side download. Built specifically for Kaggle sessions,
where /kaggle/working is wiped on every brand-new session unless you
manually "Save Version" before the old one ends (easy to forget mid-run,
and the loss is silent until the next session starts fresh with no
checkpoint). Colab doesn't need this -- its checkpoint-dir already lives on
a mounted Drive.

Uses a service account rather than interactive OAuth because a Kaggle
kernel has no browser to complete an OAuth consent screen. One-time setup
(see the "Auto-upload checkpoints to Drive" cell in run_kaggle.ipynb):
  1. Google Cloud Console -> create/select a project -> enable the
     "Google Drive API".
  2. IAM & Admin -> Service Accounts -> Create Service Account -> Keys ->
     Add Key -> JSON. Download the key file.
  3. Open the key file, copy its "client_email" value, and Share the
     target Drive folder with that email as Editor.
  4. In Kaggle: Add-ons -> Secrets -> add a secret holding the full JSON
     key file content (e.g. named GDRIVE_SERVICE_ACCOUNT_JSON).
"""

import os


def upload_checkpoint_to_drive(local_path: str, folder_id: str, service_account_json: str) -> bool:
    """Upload/overwrite `local_path` in Drive folder `folder_id`, authenticating with the service
    account credentials at `service_account_json`. Returns True on success, False on any failure
    (logged, never raised) -- a Drive hiccup must never crash a training run.

    Finds any existing Drive file with the same name in that folder and calls files.update() on its
    id (overwrite in place, same file id/link every time) instead of files.create() again, so
    repeated calls during training never pile up duplicate copies in the folder.
    """
    if not os.path.exists(local_path):
        print(f"[drive] {local_path} does not exist yet -- skipping upload.")
        return False
    try:
        from google.oauth2 import service_account as sa
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = sa.Credentials.from_service_account_file(
            service_account_json, scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        filename = os.path.basename(local_path)
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing = service.files().list(q=query, fields="files(id)", spaces="drive").execute()
        media = MediaFileUpload(local_path, resumable=False)

        matches = existing.get("files", [])
        if matches:
            service.files().update(fileId=matches[0]["id"], media_body=media).execute()
        else:
            service.files().create(body={"name": filename, "parents": [folder_id]}, media_body=media).execute()

        print(f"[drive] uploaded {local_path} -> Drive folder {folder_id}")
        return True
    except Exception as e:
        print(f"[drive] checkpoint upload skipped due to error: {e}")
        return False
