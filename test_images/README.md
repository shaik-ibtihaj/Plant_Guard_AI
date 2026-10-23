# Plant Guard sample test images

This folder contains 10 generated leaf photos for trying the upload and
inference workflow. Upload any PNG in the app's **Diagnose plant** page.

Start the backend from the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Then start the frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

Open <http://localhost:5173> and choose an image from this folder.

Files are named for their intended visual theme: healthy tomato and pepper
leaves, plus illustrative early blight, apple scab, apple black rot, grape
black rot, corn common rust, potato late blight, peach bacterial spot, and
strawberry leaf scorch examples.

Important: these are AI-generated workflow samples, not verified photographs
or ground-truth disease labels. Their visual symptoms and filenames are for
demonstration only. Do not use the model's output on these images to measure
accuracy or guide plant treatment. The checked-in model is a low-accuracy
benchmark and all results are advisory.
