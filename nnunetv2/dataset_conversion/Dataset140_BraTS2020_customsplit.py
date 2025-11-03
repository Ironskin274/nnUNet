import os
import shutil
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subdirs, save_json

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


def _read_list_file(list_path: str) -> List[str]:
    with open(list_path, 'r', encoding='utf-8') as f:
        ids = [l.strip() for l in f.readlines() if l.strip()]
    return ids


def _find_existing_file(basepath_without_ext: str) -> Tuple[str, str]:
    """
    Given a base path without extension, return (full_path, file_ending).
    Tries .nii.gz then .nii.
    """
    for ending in ('.nii.gz', '.nii'):
        p = basepath_without_ext + ending
        if os.path.isfile(p):
            return p, ending
    raise FileNotFoundError(f"File not found for base '{basepath_without_ext}' with .nii.gz/.nii")


def _copy_image_modalities(case_dir: str, case_id: str, out_images_dir: str, desired_ending: str = '.nii') -> str:
    """
    Copy T1, T1ce, T2, Flair into out_images_dir as case_id_0000..0003 with the detected extension.
    Returns detected file ending (e.g., .nii.gz or .nii).
    """
    # Channel order must be consistent with dataset_json: 0000=T1, 0001=T1ce, 0002=T2, 0003=Flair
    src_bases = {
        '0000': join(case_dir, f"{case_id}_t1"),
        '0001': join(case_dir, f"{case_id}_t1ce"),
        '0002': join(case_dir, f"{case_id}_t2"),
        '0003': join(case_dir, f"{case_id}_flair"),
    }

    for channel, src_base in src_bases.items():
        src_file, ending = _find_existing_file(src_base)
        out_file = join(out_images_dir, f"{case_id}_{channel}{desired_ending}")
        if ending == desired_ending:
            shutil.copy(src_file, out_file)
        else:
            # convert by re-saving with SimpleITK to enforce consistent ending
            img = sitk.ReadImage(src_file)
            sitk.WriteImage(img, out_file)

    return desired_ending


def _convert_and_copy_seg(seg_file_in: str, seg_file_out: str) -> None:
    """Map BraTS labels 0,1,2,4 -> nnU-Net 0,2,1,3 and save."""
    img = sitk.ReadImage(seg_file_in)
    seg = sitk.GetArrayFromImage(img)

    uniques = np.unique(seg)
    for u in uniques:
        if u not in (0, 1, 2, 4):
            raise RuntimeError(f"Unexpected label in {seg_file_in}: {u}")

    seg_new = np.zeros_like(seg, dtype=seg.dtype)
    seg_new[seg == 4] = 3
    seg_new[seg == 2] = 1
    seg_new[seg == 1] = 2

    img_out = sitk.GetImageFromArray(seg_new)
    img_out.CopyInformation(img)
    sitk.WriteImage(img_out, seg_file_out)


def _gather_validation_cases(validation_root: str) -> List[str]:
    # BraTS20 validation usually like BraTS20_Validation_XXX
    cases = subdirs(validation_root, prefix='BraTS20_Validation_', join=False)
    return cases


def main():
    # ====== Configure your source paths here ======
    brats20_training_root = \
        "/data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData"
    brats20_validation_root = \
        "/data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_ValidationData"
    train_list_path = \
        "/data/ssd2/liying/Datasets/BraTS2020/train_list.txt"
    valid_list_path = \
        "/data/ssd2/liying/Datasets/BraTS2020/valid_list.txt"

    # ====== Dataset ID and Name ======
    dataset_id = 140
    dataset_name = f"Dataset{dataset_id:03d}_BraTS2020_custom"

    # ====== Prepare nnUNet_raw structure ======
    out_base = join(nnUNet_raw, dataset_name)
    imagesTr = join(out_base, 'imagesTr')
    labelsTr = join(out_base, 'labelsTr')
    imagesTs = join(out_base, 'imagesTs')  # original validation as test set
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(imagesTs)

    # ====== Read desired splits ======
    train_ids = _read_list_file(train_list_path)
    val_ids = _read_list_file(valid_list_path)

    # Basic existence check in training root
    all_train_case_dirs = set(subdirs(brats20_training_root, prefix='BraTS20_Training_', join=False))
    missing = [cid for cid in (train_ids + val_ids) if cid not in all_train_case_dirs]
    if len(missing) > 0:
        raise FileNotFoundError(f"Missing cases in TrainingData: {missing[:5]} ... total {len(missing)}")

    # ====== Copy TRAIN/VAL splits (images+labels) with validation ======
    detected_file_ending = '.nii'
    copied_train: List[str] = []
    copied_val: List[str] = []
    skipped: List[Tuple[str, str]] = []  # (case_id, reason)

    def _copy_case(cid: str, split_name: str):
        nonlocal detected_file_ending
        src_case_dir = join(brats20_training_root, cid)
        try:
            ending = _copy_image_modalities(src_case_dir, cid, imagesTr, desired_ending=detected_file_ending)
        except FileNotFoundError as e:
            skipped.append((cid, f"missing modality: {e}"))
            return
        try:
            seg_in, _ = _find_existing_file(join(src_case_dir, f"{cid}_seg"))
        except FileNotFoundError as e:
            skipped.append((cid, f"missing seg: {e}"))
            # remove possibly copied images to keep consistency
            for ch in ('0000', '0001', '0002', '0003'):
                for ext in ('.nii.gz', '.nii'):
                    p = join(imagesTr, f"{cid}_{ch}{ext}")
                    if os.path.isfile(p):
                        os.remove(p)
            return
        seg_out = join(labelsTr, f"{cid}{detected_file_ending}")
        _convert_and_copy_seg(seg_in, seg_out)
        if split_name == 'train':
            copied_train.append(cid)
        else:
            copied_val.append(cid)

    for cid in train_ids:
        _copy_case(cid, 'train')

    for cid in val_ids:
        _copy_case(cid, 'val')

    # ====== Copy TEST split from original ValidationData (images only) ======
    test_case_ids = _gather_validation_cases(brats20_validation_root)
    for cid in test_case_ids:
        src_case_dir = join(brats20_validation_root, cid)
        ending = _copy_image_modalities(src_case_dir, cid, imagesTs, desired_ending=detected_file_ending)

    if detected_file_ending is None:
        raise RuntimeError("Could not determine file ending from source files.")

    # ====== Post-copy strict validation & sanitization ======
    def _has_complete_case(case_id: str) -> bool:
        img_ok = all(os.path.isfile(join(imagesTr, f"{case_id}_{ch}{detected_file_ending}")) for ch in ('0000', '0001', '0002', '0003'))
        lbl_ok = os.path.isfile(join(labelsTr, f"{case_id}{detected_file_ending}"))
        return img_ok and lbl_ok

    incomplete_cases: List[str] = []
    for cid in list(copied_train) + list(copied_val):
        if not _has_complete_case(cid):
            incomplete_cases.append(cid)
            # remove partial leftovers
            for ch in ('0000', '0001', '0002', '0003'):
                p = join(imagesTr, f"{cid}_{ch}{detected_file_ending}")
                if os.path.isfile(p):
                    os.remove(p)
            p = join(labelsTr, f"{cid}{detected_file_ending}")
            if os.path.isfile(p):
                os.remove(p)
            if cid in copied_train:
                copied_train.remove(cid)
            if cid in copied_val:
                copied_val.remove(cid)

    # ====== Generate dataset.json ======
    # Use actually copied count to avoid integrity mismatch
    num_training_cases = len(copied_train) + len(copied_val)

    generate_dataset_json(
        out_base,
        channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
        labels={
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': (3,)
        },
        num_training_cases=num_training_cases,
        file_ending=detected_file_ending,
        regions_class_order=(1, 2, 3),
        license='BraTS 2020',
        reference='MICCAI BraTS 2020',
        dataset_release='custom split'
    )

    # ====== Write splits_final.json to nnUNet_preprocessed (5 identical folds) ======
    pp_out_dir = join(nnUNet_preprocessed, dataset_name)
    maybe_mkdir_p(pp_out_dir)
    # only include cases we actually copied
    single_split = {
        'train': copied_train,
        'val': copied_val
    }
    splits = [single_split] * 5
    save_json(splits, join(pp_out_dir, 'splits_final.json'), sort_keys=False)

    print(f"Prepared {dataset_name} at: {out_base}")
    print(f"Training cases (requested -> copied -> kept): {len(train_ids)} -> {len(copied_train) + len(incomplete_cases)} -> {len(copied_train)} | "
          f"Validation cases: {len(val_ids)} -> {len(copied_val) + len([c for c in incomplete_cases if c in val_ids])} -> {len(copied_val)} | Test cases: {len(test_case_ids)}")
    if skipped:
        print(f"Skipped {len(skipped)} cases due to missing files (showing up to 10):")
        for cid, reason in skipped[:10]:
            print(f"  - {cid}: {reason}")
    if incomplete_cases:
        print(f"Removed {len(incomplete_cases)} incomplete cases after validation (showing up to 10):")
        for cid in incomplete_cases[:10]:
            print(f"  - {cid}")
    print(f"splits_final.json written to: {join(pp_out_dir, 'splits_final.json')}")


if __name__ == '__main__':
    main()


