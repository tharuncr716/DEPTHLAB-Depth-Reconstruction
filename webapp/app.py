
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

WEBAPP_DIR = BASE_DIR / "webapp"
MODEL_DIR = BASE_DIR / "models"
CLASSICAL_DIR = BASE_DIR / "classical"

UPLOAD_DIR = WEBAPP_DIR / "static" / "uploads"
RESULT_DIR = WEBAPP_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_depth_model.pth"


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(CLASSICAL_DIR))

from depth_cnn import DepthRefinementCNN
from classical_methods import apply_methods


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "png", "jpg", "jpeg", "tif", "tiff", "bmp"
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DEPTHLAB device:", DEVICE)


# ============================================================
# LOAD TRAINED CNN MODEL
# ============================================================

model = DepthRefinementCNN().to(DEVICE)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("CNN model loaded successfully.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def read_rgb_image(file_path):
    """
    Read an RGB image and return RGB uint8 format.
    """

    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to read the uploaded RGB image.")

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def read_depth_image(file_path):
    """
    Read depth image while preserving its original bit depth.

    Supported conventions:
    - 16-bit integer PNG: depth in millimetres.
    - 8-bit grayscale: interpreted as normalized relative depth.
    - Floating-point depth: assumed to be in metres.
    """

    depth = cv2.imread(
        str(file_path),
        cv2.IMREAD_UNCHANGED
    )

    if depth is None:
        raise ValueError("Unable to read the uploaded depth map.")

    # Convert colour depth image to grayscale if necessary.
    if depth.ndim == 3:

        if depth.shape[2] == 4:
            depth = cv2.cvtColor(
                depth,
                cv2.COLOR_BGRA2GRAY
            )

        else:
            depth = cv2.cvtColor(
                depth,
                cv2.COLOR_BGR2GRAY
            )

    # Convert to floating point.
    original_dtype = depth.dtype
    depth = depth.astype(np.float32)

    if not np.all(np.isfinite(depth)):
        raise ValueError("Depth map contains invalid values.")

    if np.max(depth) <= 0:
        raise ValueError("Depth map contains no valid positive depth values.")

    # Convert 16-bit millimetres into metres.
    if original_dtype in (np.uint16, np.int16):
        depth = depth / 1000.0

    # Convert 8-bit normalized depth into an approximate
    # 0.1 to 10 metre range.
    elif original_dtype == np.uint8:
        depth = 0.1 + (depth / 255.0) * 9.9

    # Floating-point depth is assumed to already be in metres.

    depth = np.maximum(depth, 0.1)

    return depth


def resize_depth_map(depth, target_width, target_height):
    """
    Automatically resize uploaded depth to the dimensions
    required by the selected upsampling factor.
    """

    original_height, original_width = depth.shape[:2]

    if (
        original_width == target_width
        and original_height == target_height
    ):
        return depth

    print(
        f"Automatic depth resizing: "
        f"{original_width}x{original_height} -> "
        f"{target_width}x{target_height}"
    )

    # Use area interpolation when reducing resolution.
    if (
        target_width < original_width
        or target_height < original_height
    ):
        interpolation = cv2.INTER_AREA

    else:
        interpolation = cv2.INTER_CUBIC

    resized = cv2.resize(
        depth,
        (target_width, target_height),
        interpolation=interpolation
    )

    return resized.astype(np.float32)


def bicubic_upsample(low_depth, width, height):
    """
    Generate the bicubic depth input for CNN refinement.
    """

    return cv2.resize(
        low_depth,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)



def run_cnn(rgb_image, bicubic_depth):
    """
    Run the trained CNN model using separate RGB and depth inputs.
    """

    rgb_normalized = (
        rgb_image.astype(np.float32) / 255.0
    )

    rgb_tensor = torch.from_numpy(
        rgb_normalized.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(DEVICE)

    depth_tensor = torch.from_numpy(
        bicubic_depth
    ).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    model.eval()

    with torch.no_grad():

        # IMPORTANT: Pass RGB and depth separately.
        prediction = model(
            rgb_tensor,
            depth_tensor
        )

    prediction = (
        prediction.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    prediction = np.maximum(prediction, 0.1)

    return prediction.astype(np.float32)



def run_classical(rgb_image, low_depth):
    """
    Run the classical depth refinement methods.

    Returns the Joint Bilateral Filter result.
    """

    methods = apply_methods(
        rgb_image,
        low_depth
    )

    if "Joint Bilateral" not in methods:
        raise ValueError(
            "Joint Bilateral result was not returned by classical_methods.py"
        )

    return np.asarray(
        methods["Joint Bilateral"],
        dtype=np.float32
    )


def run_hybrid(cnn_depth, jbf_depth, factor):
    """
    Combine CNN and Joint Bilateral predictions.

    Fusion weights were selected using validation data.
    """

    alpha_values = {
        2: 0.75,
        4: 0.50,
        8: 0.00
    }

    alpha = alpha_values[factor]

    hybrid_depth = (
        alpha * cnn_depth
        + (1.0 - alpha) * jbf_depth
    )

    return np.maximum(
        hybrid_depth,
        0.1
    ).astype(np.float32)


def save_depth_visualization(depth, output_path):
    """
    Convert numerical depth into a colour visualization
    for display in the web application.

    The original depth values are also saved separately
    as a 32-bit floating-point TIFF.
    """

    depth = np.nan_to_num(
        depth,
        nan=0.1,
        posinf=0.1,
        neginf=0.1
    )

    minimum = float(np.min(depth))
    maximum = float(np.max(depth))

    if maximum > minimum:

        normalized = (
            (depth - minimum)
            / (maximum - minimum)
            * 255
        )

    else:
        normalized = np.zeros_like(depth)

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    colored = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_TURBO
    )

    cv2.imwrite(
        str(output_path),
        colored
    )


# ============================================================
# MAIN ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # 1. GET UPLOADED FILES
            # ------------------------------------------------

            rgb_file = request.files.get("rgb")
            depth_file = request.files.get("depth")

            if not rgb_file or not rgb_file.filename:
                raise ValueError("Please upload an RGB image.")

            if not depth_file or not depth_file.filename:
                raise ValueError("Please upload a depth map.")

            if not allowed_file(rgb_file.filename):
                raise ValueError("Unsupported RGB image format.")

            if not allowed_file(depth_file.filename):
                raise ValueError("Unsupported depth image format.")


            # ------------------------------------------------
            # 2. GET SCALE AND METHOD
            # ------------------------------------------------

            factor = int(
                request.form.get("factor", 4)
            )

            method = request.form.get(
                "method",
                "hybrid"
            ).lower()

            if factor not in [2, 4, 8]:
                raise ValueError(
                    "Invalid scale. Select 2x, 4x or 8x."
                )

            if method not in [
                "classical",
                "cnn",
                "hybrid"
            ]:
                raise ValueError(
                    "Invalid reconstruction method."
                )


            # ------------------------------------------------
            # 3. SAVE UPLOADED FILES
            # ------------------------------------------------

            unique_id = uuid.uuid4().hex[:12]

            rgb_filename = (
                unique_id + "_" +
                secure_filename(rgb_file.filename)
            )

            depth_filename = (
                unique_id + "_" +
                secure_filename(depth_file.filename)
            )

            rgb_path = UPLOAD_DIR / rgb_filename
            depth_path = UPLOAD_DIR / depth_filename

            rgb_file.save(str(rgb_path))
            depth_file.save(str(depth_path))


            # ------------------------------------------------
            # 4. READ INPUT IMAGES
            # ------------------------------------------------

            rgb = read_rgb_image(rgb_path)
            depth = read_depth_image(depth_path)

            rgb_height, rgb_width = rgb.shape[:2]

            print("\n========== DEPTHLAB PROCESSING ==========")

            print(
                "RGB resolution:",
                rgb_width,
                "x",
                rgb_height
            )

            print(
                "Uploaded depth resolution:",
                depth.shape[1],
                "x",
                depth.shape[0]
            )


            # ------------------------------------------------
            # 5. AUTOMATIC DIMENSION HANDLING
            # ------------------------------------------------

            if (
                rgb_width % factor != 0
                or rgb_height % factor != 0
            ):
                raise ValueError(
                    f"RGB dimensions {rgb_width}x{rgb_height} "
                    f"must be divisible by {factor}."
                )

            expected_width = rgb_width // factor
            expected_height = rgb_height // factor

            original_depth_height, original_depth_width = (
                depth.shape[:2]
            )

            depth = resize_depth_map(
                depth,
                expected_width,
                expected_height
            )

            print(
                "Adjusted depth resolution:",
                depth.shape[1],
                "x",
                depth.shape[0]
            )

            print(
                "Expected depth resolution:",
                expected_width,
                "x",
                expected_height
            )


            # ------------------------------------------------
            # 6. BICUBIC UPSAMPLING
            # ------------------------------------------------

            bicubic_depth = bicubic_upsample(
                depth,
                rgb_width,
                rgb_height
            )


            # ------------------------------------------------
            # 7. CLASSICAL JOINT BILATERAL FILTER
            # ------------------------------------------------

            jbf_depth = None
            cnn_depth = None

            if method in ["classical", "hybrid"]:

                print("Running Joint Bilateral Filter...")

                jbf_depth = run_classical(
                    rgb,
                    depth
                )

                if jbf_depth.shape != (rgb_height, rgb_width):
                    raise ValueError(
                        "Classical output dimensions do not match RGB image."
                    )


            # ------------------------------------------------
            # 8. CNN REFINEMENT
            # ------------------------------------------------

            if method in ["cnn", "hybrid"]:

                print("Running CNN refinement...")

                cnn_depth = run_cnn(
                    rgb,
                    bicubic_depth
                )

                if cnn_depth.shape != (rgb_height, rgb_width):
                    raise ValueError(
                        "CNN output dimensions do not match RGB image."
                    )


            # ------------------------------------------------
            # 9. SELECT FINAL RECONSTRUCTION
            # ------------------------------------------------

            if method == "classical":

                final_depth = jbf_depth

                method_name = "Classical - Joint Bilateral"

            elif method == "cnn":

                final_depth = cnn_depth

                method_name = "Deep Learning - CNN"

            else:

                final_depth = run_hybrid(
                    cnn_depth,
                    jbf_depth,
                    factor
                )

                method_name = "Hybrid - CNN + Joint Bilateral"


            # ------------------------------------------------
            # 10. SAVE OUTPUT
            # ------------------------------------------------

            output_filename = (
                f"depth_{method}_{factor}x_{unique_id}.png"
            )

            output_path = RESULT_DIR / output_filename

            save_depth_visualization(
                final_depth,
                output_path
            )

            # Also save numerical depth as a 32-bit TIFF.
            raw_filename = (
                f"depth_{method}_{factor}x_{unique_id}.tiff"
            )

            raw_path = RESULT_DIR / raw_filename

            cv2.imwrite(
                str(raw_path),
                final_depth.astype(np.float32)
            )


            # ------------------------------------------------
            # 11. PREPARE RESULT FOR HTML
            # ------------------------------------------------

            result = {

                "method": method_name,

                "factor": factor,

                "input": url_for(
                    "static",
                    filename=f"uploads/{rgb_filename}"
                ),

                "output": url_for(
                    "static",
                    filename=f"results/{output_filename}"
                ),

                "raw_output": url_for(
                    "static",
                    filename=f"results/{raw_filename}"
                )

            }

            print("Reconstruction completed successfully.")

            print("Final output:", output_path)

            print("Raw depth:", raw_path)

            print("=========================================\n")


        except Exception as e:

            error = str(e)

            print("DEPTHLAB ERROR:", error)


    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("       DEPTHLAB AI STUDIO")
    print("===================================")
    print("Device:", DEVICE)
    print("Server: http://127.0.0.1:5000")
    print("===================================\n")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )