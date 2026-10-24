# E5a launcher-copy provenance (written 2026-10-22T21:17:07+05:30, before submission)

Copy: external/Quantized-Reasoning-Models_e5a (tar copy of external/Quantized-Reasoning-Models excluding .git, __pycache__ and third-party; third-party is a symlink to the original, so lighteval/vLLM sources are identical).
Single change (lighteval_custom/models/vllm/vllm_model.py line 280); diff sha256[:16] = 91f8554c322483e7:
```diff
--- external/Quantized-Reasoning-Models/lighteval_custom/models/vllm/vllm_model.py	2026-10-13 05:36:23.000000000 +0530
+++ external/Quantized-Reasoning-Models_e5a/lighteval_custom/models/vllm/vllm_model.py	2026-10-22 21:16:04.000000000 +0530
@@ -277,7 +277,7 @@
             num_samples = dataset[0].num_samples
 
             context = [c.context for c in dataset]
-            tokenized = self.tokenizer(context, add_special_tokens=self.add_special_tokens)
+            tokenized = self.tokenizer(context, add_special_tokens=False if os.environ.get("R2_ADD_SPECIAL_TOKENS") == "0" else self.add_special_tokens)
 
             # The main question for this step is the following:
             # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
```
diff -r (excluding __pycache__, .git, third-party symlink) between the original tree and the copy shows only that file, only that line.
Original tree hash with the manifest recipe, excluding the two new E5a scripts: 00ff9fe5b1d65b3b = the code_tree_sha256_16 recorded in the E2/E3/X1/X3/E1c manifests (original tree unchanged).
Copy tree hash with the same recipe (copy path substituted, new scripts included): 1da3552168caf5c1. lighteval_custom-only hash: original 2e362e2a1b713d31, copy ce4dda0700b65828.
Manifests of E5a cells record code_tree_sha256_16 of the ORIGINAL path (write_manifest.py is unchanged and hashes external/Quantized-Reasoning-Models); the copy's hash and the diff hash are recorded in the manifest env_vars as R2_E5A_COPY_CODE_TREE_SHA256_16 and R2_E5A_ONE_LINE_DIFF_SHA256_16, and QRM_REPO_DIR shows the copy path.
Harness differences vs the E2/E3/X3 harness: scripts/rev2/run_cells_e5a.sh = run_cells.sh with (1) QRM_REPO_DIR pointing at the copy, (2) the extra first-cell gate item scripts/rev2/e5a_ids_check.py. verify_cell.py and first_cell_check.py are unchanged.
