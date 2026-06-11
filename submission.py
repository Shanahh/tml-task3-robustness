import os
import sys
import requests

"""
Submission script for the Robustness task.

Submission Requirements (read carefully to avoid automatic rejection):

1. FILE FORMAT
----------------
- The file must be a PyTorch state dict saved as a .pt file.
- Save only the state dict, not the full model instance:
      torch.save(model.state_dict(), "model.pt")  # correct
      torch.save(model, "model.pt")               # wrong

2. MODEL ARCHITECTURE
----------------------
- You must specify the model architecture using the model-name field.
- Allowed values: resnet18, resnet34, resnet50
- The architecture must match the state dict you are submitting.

3. MODEL REQUIREMENTS
----------------------
- Input shape must be (1, 3, 32, 32)
- Output shape must be (1, 9)
- The final fc layer must be replaced to output 9 classes

4. EVALUATION
--------------
- Your model must achieve clean accuracy greater than 50% to be accepted.
- Submissions below this threshold will be automatically rejected.
- Score = 0.5 * clean accuracy + 0.5 * robustness accuracy

5. TECHNICAL LIMITS
--------------------
- Only one submission per group every 60 minutes.
- If your submission fails due to an error on your side, the cooldown is 2 minutes.

Your submission will fail if:
- The file is not a valid .pt state dict
- The model-name does not match the submitted state dict
- The output shape is not (1, 9)
- The input shape is not (1, 3, 32, 32)
- Clean accuracy is below 50%
"""

BASE_URL = "http://34.63.153.158"
API_KEY = "ef283bd434803248eee23a3b016499ae"  # replace with your actual API key

MODEL_PATH = "/home/atml_team052/tml-task3-robustness/model.pt"  # replace with your actual model path
MODEL_NAME = "resnet18"  # replace with your actual model architecture - resnet18, resnet34, or resnet50

SUBMIT = True  # set to True to enable submission

TASK_ID = "03-robustness" # donot change


def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)


if SUBMIT:
    if not os.path.isfile(MODEL_PATH):
        die(f"File not found: {MODEL_PATH}")

    try:
        with open(MODEL_PATH, "rb") as f:
            files = {"file": (os.path.basename(MODEL_PATH), f, "application/x-pytorch")}
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                data={"model_name": MODEL_NAME},
                # timeout=(10, 520),
            )

        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        print("Successfully submitted.")
        print("Server response:", body)

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)
