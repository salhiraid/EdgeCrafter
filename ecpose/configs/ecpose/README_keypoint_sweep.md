# ECPose Keypoint Config Sweep

This folder includes keypoint-focused variants for the S and M COCO pose configs.

Recommended order:

1. `ecpose_s_coco_kpt_01_mild_keypoint_boost.yml` or `ecpose_m_coco_kpt_01_mild_keypoint_boost.yml`
2. `ecpose_s_coco_kpt_02_oks_focus.yml` or `ecpose_m_coco_kpt_02_oks_focus.yml`
3. `ecpose_s_coco_kpt_04_clean_finetune_no_heavy_aug.yml` or `ecpose_m_coco_kpt_04_clean_finetune_no_heavy_aug.yml`
4. `ecpose_s_coco_kpt_03_strong_keypoint_localization.yml` or `ecpose_m_coco_kpt_03_strong_keypoint_localization.yml`
5. `ecpose_s_coco_kpt_05_more_queries_crowded.yml` or `ecpose_m_coco_kpt_05_more_queries_crowded.yml`

The `00_baseline_explicit` files keep the official ECPose balance explicit:
`loss_vfl: 2.0`, `loss_keypoints: 10.0`, and `loss_oks: 4.0`.

The existing `ecpose_s_coco_kpt_boost.yml` and `ecpose_m_coco_kpt_boost.yml`
are simple aliases of the mild keypoint boost recipe.

