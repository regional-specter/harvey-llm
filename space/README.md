---
title: Harvey LLM Chat
emoji: 👔
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
---

Chat with the fine-tuned Harvey Specter model.

## Recommended: T4 GPU hardware

In **Settings → Hardware → GPU → T4 small**. The model loads once at startup — no ZeroGPU quota, fast replies. Pause the Space when not in use to stop billing (~$0.60/hr while running).

## Free tier: ZeroGPU

Works on CPU basic hardware but has a **daily GPU time quota**. Log in with your Hugging Face account on the Space page for more quota. A 7B model burns through free quota quickly — T4 is strongly recommended for regular use.
