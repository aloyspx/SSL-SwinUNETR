import argparse
import glob
import os
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def resample_image(itk_image, out_spacing):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))),
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(-1000)

    resample.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resample.Execute(itk_image)

    return resampled_image


def process_image(ct_filename, in_dir, brain_dir, out_dir):
    ct_path = os.path.join(in_dir, ct_filename)
    ct_image = sitk.ReadImage(ct_path)

    ct_resampled_image = resample_image(ct_image, [0.5, 0.5, 5.0])

    ct_hu_min, ct_hu_max = 0, 80
    clipped_ct_data = np.clip(sitk.GetArrayFromImage(ct_resampled_image), ct_hu_min, ct_hu_max)

    ct_data_norm = (clipped_ct_data - np.min(clipped_ct_data)) / (np.max(clipped_ct_data) - np.min(clipped_ct_data))

    new_img = sitk.GetImageFromArray(ct_data_norm)
    new_img.SetSpacing(ct_resampled_image.GetSpacing())
    new_img.SetOrigin(ct_resampled_image.GetOrigin())
    new_img.SetDirection(ct_resampled_image.GetDirection())

    output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(ct_filename))[0] + ".nii.gz")
    sitk.WriteImage(new_img, output_path)


def preprocess_nifti_images(in_dir, brain_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    ct_files = glob.glob(f"{in_dir}/*.nii.gz")
    args = [(filename, in_dir, brain_dir, out_dir) for filename in ct_files]

    with Pool() as pool:
        list(tqdm(pool.starmap(process_image, args), total=len(args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Nifti Images")
    parser.add_argument("--in_dir", type=str, help="Input directory containing all the images (CT and PET, mixed)")
    parser.add_argument("--brain_dir", type=str, help="Brain segmentation directory")
    parser.add_argument("--out_dir", type=str, help="Directory to save the images")
    args = parser.parse_args()

    preprocess_nifti_images(args.in_dir, args.brain_dir, args.out_dir)
