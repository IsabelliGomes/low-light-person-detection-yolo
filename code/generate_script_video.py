import cv2
import numpy as np

def apply_filters(
    video_path,
    output_path,
    exposure=1.0,
    brightness=0,
    contrast=1.0,
    shadow=-0.5,
    gamma=1.0,
    vinheta_size=200,
    vinheta_intensity=0.5,
    bilateral_filter=True,
    equalize_hist=True,
    gaussian_blur=True
):
    """
    Apply lighting enhancement filters to a video to make it closer to an infrared-like image.

    Args:
        video_path (str): Input video path.
        output_path (str): Output video path.
        exposure (float): Exposure factor (1.0 = no change).
        brightness (int): Brightness value (0 = no change).
        contrast (float): Contrast factor (1.0 = no change).
        shadow (float): Shadow intensity (-1 to 1, default -0.5).
        gamma (float): Gamma correction (1.0 = no change).
        vinheta_size (int): Vignette size (larger means smoother edges).
        vinheta_intensity (float): Vignette intensity (0.0 to 1.0, default 0.5).
        bilateral_filter (bool): Apply bilateral filter to smooth noise while preserving details.
        equalize_hist (bool): Apply histogram equalization to improve contrast.
        gaussian_blur (bool): Apply Gaussian blur to reduce excessive grain.
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to save the video

    # Configure video output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Adjust exposure and brightness
        frame = cv2.convertScaleAbs(frame, alpha=exposure, beta=brightness)

        # Adjust contrast
        frame = cv2.addWeighted(frame, contrast, frame, 0, 128 * (1 - contrast))

        # Adjust shadows (reduce dark tones)
        shadow_mask = np.full_like(frame, 50)  # More granular shadow adjustment
        frame = cv2.addWeighted(frame, 1, shadow_mask, shadow, 0)

        # Gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        frame = cv2.LUT(frame, table)

        # Apply bilateral filter to preserve details and reduce noise
        if bilateral_filter:
            frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # Apply histogram equalization to improve contrast
        if equalize_hist:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = cv2.equalizeHist(l)
            lab_eq = cv2.merge((l_eq, a, b))
            frame = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Apply Gaussian blur to smooth grain
        if gaussian_blur:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Create vignette mask
        rows, cols = frame.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, vinheta_size)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, vinheta_size)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        vinheta_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply vignette
        vinheta = cv2.merge([vinheta_mask] * 3)  # Create a 3-channel mask
        frame = cv2.addWeighted(frame, 1 - vinheta_intensity, vinheta, vinheta_intensity, 0)

        # Save processed frame
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage example
video_input = # YOUR INPUT VIDEO PATH HERE
video_output = # YOUR OUTPUT VIDEO PATH HERE

apply_filters(
    video_path=video_input,
    output_path=video_output,
    exposure=1.75,
    brightness=100,
    contrast=0.75,
    shadow=-0.25,
    gamma=1.2,
    vinheta_size=175,
    vinheta_intensity=0.1,
    bilateral_filter=True,
    equalize_hist=True,
    gaussian_blur=True
)
