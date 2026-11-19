# Training the emotion CNN on a free GPU

On a laptop CPU, `python -m src.train_emotion --epochs 40` takes about **3 hours**.
On a free cloud T4 GPU it takes about **10–20 minutes**.
This guide covers Google Colab (no phone verification needed) and Kaggle.

## What you need

- `emomusic_src.zip`: the `src/` folder zipped. Create it from the `EmoMusic/` folder:

  ```bash
  python -c "import zipfile, pathlib; z = zipfile.ZipFile('emomusic_src.zip', 'w', zipfile.ZIP_DEFLATED); [z.write(f, f.as_posix()) for f in sorted(pathlib.Path('src').glob('*.py'))]"
  ```

  Python's `zipfile` writes `src/...` paths with forward slashes. Windows PowerShell 5.1's
  `Compress-Archive` writes backslashes, which Linux does not unpack as a folder.
- The FER-2013 dataset ([msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013) on Kaggle).
  The steps below download it for you.

## Option A: Google Colab

1. Open <https://colab.research.google.com> and create a new notebook.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Paste this into one cell and run it:

   ```python
   # upload the code (choose emomusic_src.zip)
   from google.colab import files
   files.upload()
   !unzip -q -o emomusic_src.zip

   # download FER-2013 from Kaggle
   import kagglehub
   path = kagglehub.dataset_download("msambare/fer2013")
   print(path)

   # train (Colab has 2 CPU cores, used for data augmentation)
   !python -m src.train_emotion --data "$path" --epochs 40 --workers 2
   ```

   The first output line must say `device: cuda`. If it says `cpu`, the GPU is not on (go back to step 2).
4. When training ends, download the results **right away**, because Colab deletes them when the session closes:

   ```python
   files.download("models/emotion_cnn.pth")
   files.download("output/emotion_confusion_matrix.png")
   files.download("output/emotion_training_curves.png")
   ```

5. Put `emotion_cnn.pth` in your local `EmoMusic/models/` folder.

## Option B: Kaggle

Kaggle allows GPUs only after the account is **phone-verified** (kaggle.com → Settings → Phone verification;
choose the right country code, or the SMS goes to the wrong country).

1. Create a notebook. **Settings → Accelerator → GPU T4 x2.**
2. **Add Input → Upload → New Dataset**: upload `emomusic_src.zip` (Kaggle unzips it).
3. **Add Input** → search `fer2013` → click **+** on "FER-2013" by Manas Sambare.
4. Run:

   ```python
   !ls /kaggle/input/*            # check the real folder names
   !cp -r /kaggle/input/emomusic-src/src /kaggle/working/src
   %cd /kaggle/working
   !python -m src.train_emotion --data /kaggle/input/fer2013 --epochs 40 --workers 4
   ```

   You need the copy because `/kaggle/input` is read-only, and the script saves the model next to the code.
   `--data` must point to the folder that contains `train/` and `test/`.
5. Download `models/emotion_cnn.pth` from the **Output** panel and put it in `EmoMusic/models/`.

## Reading the log

Each epoch prints a line like:

```
ep  12  loss 0.912/0.987  acc 0.661/0.684  18s  * saved
```

- `loss train/val` and `acc train/val`: the target is **val accuracy of about 0.65–0.75**.
- `* saved`: the best model so far was written to `models/emotion_cnn.pth`.
- Validation uses the FER-2013 test split. Choosing the best epoch on it makes the reported accuracy
  slightly optimistic; mention this in the report.

If you train both locally and in the cloud, keep the model with the higher `best val acc`.
