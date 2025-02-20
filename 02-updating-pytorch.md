

---

### **📄 `docs/02-updating-pytorch.md`**  

```markdown
# Updating PyTorch and Dependencies  

Updating PyTorch and dependencies is **optional** and may break compatibility due to deprecated APIs.  

## 📌 Step 1: Check Current Versions  
From `requirements.txt`, the current versions used in DiffAE are:  
- `torch==1.8.1`  
- `torchvision` (latest compatible version)  
- `pytorch-lightning==1.4.5`  
- `torchmetrics==0.5.0`  

To check your installed PyTorch version:  
```sh
python -c "import torch; print(torch.__version__)"
```

## 📌 Step 2: Update to Latest Versions (If Needed)  
If you want to upgrade PyTorch for better performance, use the following commands:  

For CUDA 12.1:  
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning torchmetrics
```

For CPU-only version:  
```sh
pip install torch torchvision torchaudio
pip install pytorch-lightning torchmetrics
```

## 📌 Step 3: Verify Installation  
After updating, verify the installed versions:  
```sh
python -c "import torch; print(torch.__version__)"
python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"
```

## 📌 Final Recommendation  
- If you **need stability**, **stick with the current versions** from `requirements.txt`.  
- If you want **better performance**, **update and test** for compatibility.  

🚀 Now, PyTorch is updated and ready for use!  
```

---

