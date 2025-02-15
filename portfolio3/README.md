## TeSO - Tech Stack Optimizer

### Introduction
TeSO (Tech Stack Optimizer) is a Gen-AI-powered tool designed to analyze a project's current architecture and recommend an optimized tech stack. By leveraging a fine-tuned LLM model, TeSO provides intelligent recommendations that enhance system compatibility, scalability, and maintainability while also considering organizational constraints.

### How It Works
![](./images/TeSo-Workflow.png)

The user provides the technical architecture as a diagram (as code). The prompt can be customized to include additional information for optimization. The inputs are processed by a **fine-tuned LLaMA 3.2 1B Instruct model**. The model generates recommendations based with detailed explanations and reasons.

### Dataset
A synthetic dataset consisting of 61 training and 5 validation samples was created using ChatGPT and NotebookLM. The dataset consists of diagrams as code (using the `diagrams` Python package) as input, with improvement recommendations as output. System design references were uploaded to ChatGPT and NotebookLM to ensure that the generated output aligned with the book rather than providing general recommendations from an LLM. The diagram as code was collected from GitHub.

The dataset can be found [here](https://huggingface.co/datasets/gokul-pv/TeSo-Dataset)

### Training the Model

We fine-tuned the 4-bit version of the **LLaMA 3.2-1B Instruct** model using Hugging Face, the Unsloth library, and LoRA. The model was trained for 3 epochs with a learning rate of 2e-5 and a linear schedule. Unsloth’s `train_on_responses_only` was used to train only on assistant responses while ignoring the loss on user inputs. Finally, the model was saved in 16-bit precision. The trained model can be found [here](https://huggingface.co/gokul-pv/Llama-3.2-1B-Instruct-16bit-TeSO).


<table>
  <tr>
    <th>grad_norm</th>
    <th>learning_rate</th>
    <th>loss</th>
  </tr>
  <tr>
    <td><img src="./images/train_grad_norm.svg" alt="grad_norm" width="300"></td>
    <td><img src="./images/train_learning_rate.svg" alt="learning_rate" width="300"></td>
    <td><img src="./images/train_loss.svg" alt="loss" width="300"></td>
  </tr>
</table>


The training notebook can be found [here](./unsloth-llama.ipynb).

### Demo
Check out the live demo of TeSO on Hugging Face Spaces:  
[TeSO Demo](https://huggingface.co/spaces/gokul-pv/TeSo)

As the app is running on a CPU in Hugging Face Spaces, the response time may be slow. You can run the app locally by cloning the repository and executing `gradio app.py` for CPU inference or `gradio inference.py` for GPU inference.

The PDF of the presentation can be found [here](./TeSo_PPT.pdf).

---

Let TeSO guide you to the best technology stack for your next project!
