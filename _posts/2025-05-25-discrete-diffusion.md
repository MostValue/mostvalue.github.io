---
title: "Fine-Tuning Forgetfulness: Concept Erasure in Discrete Diffusion Models"
date: 2025-05-25
excerpt: "How to surgically erase concepts from generative models without retraining from scratch."
collection: blog
permalink: /blog/discrete_diffusion/concept_erasure
excerpt_separator: <!--more-->
toc: true
toc_label: "Concept Erasure"
toc_icon: "eraser"
tags:
    - Diffusion Models
    - AI Safety
    - Machine Learning
    - Concept Erasure
---

# Erasing Concepts from Discrete Diffusion Models

*Work done as part of the Technical Alignment Research Accelerator (TARA) Cohort 2025, AI Safety Australia*

Background:  
AI models, especially large generative ones, are incredibly powerful. They can create stunning art, write coherent text, and even design proteins. The issue is that, when trained with a large corpus of data, the models can learn concepts that are harmful such as copyrighted material, harmful biases, or private information. A standard solution is to retrain the model from scratch on a sanitized dataset, a process that can be astronomically expensive and time-consuming or to institute input guardrails. This led me to ask - can we teach a model to *forget*? 

This post documents my journey into the world of "concept erasure," specifically within discrete diffusion models as part of a short **2 week** project (where there definitely wasn't enough time).

This project pushed my understanding of generative AI and involved navigating complex math, some late nights debugging, and ultimately, a kind of rewarding resolution. I'll walk you through how I built a discrete diffusion model, experimented with different ways to make it unlearn a specific concept, and implemented techniques from ML literature. Hopefully this is somewhat interesting for anyone interested in AI safety, model control, or the inner workings of generative models!

---

### The Challenge: Making a Model Forget the Number 9 (as a toy problem for concept erasure)

My goal was clear and concrete:
1. Train a generative model on the full MNIST dataset, which contains thousands of images of handwritten digits (0 through 9).
2. After the model is fully trained, select one class—the digit '9'—and erase it from the model's "memory."
3. The final, edited model should no longer be able to generate the digit '9', but its ability to generate digits 0 through 8 should remain completely intact.

The project focused on **discrete diffusion models**, a class of models that are a natural fit for data that exists in distinct categories, like pixels in a simple image, words in a language, or amino acids in a protein. Unlike their continuous counterparts, they work with fixed states, which presents unique challenges and opportunities for manipulation.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/rf-diffusion.png" alt="RF Diffusion Output" style="max-width: 100%; height: auto;">
        <br>
        <small><b>RF Diffusion:</b> Discrete diffusion for protein design [cite: rf-diffusion].</small>
    </div>
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/gemeni-diffusion.png" alt="Gemeni Diffusion Output" style="max-width: 100%; height: auto;">
        <br>
        <small><b>Gemeni Diffusion:</b> Discrete diffusion for molecular generation.</small>
    </div>
</div>

> **Why discrete diffusion?**  
> Discrete diffusion models are especially relevant for domains where data is naturally categorical—such as pixels, words, or amino acids. Unlike continuous models, they can directly handle the inherent structure of these datasets, making them powerful tools for applications like protein design (RF Diffusion [cite: rf-diffusion]), molecular generation and text generation (Gemeni Diffusion).

---

## My Approach

My first step was to get a working model. I adapted an excellent [open-source implementation](https://github.com/cloneofsimo/d3pm/tree/main) of a discrete diffusion model and trained it on MNIST, discretising the grayscale pixel values into 32 classes. 


> **Diffusion Background**  
> The core idea behind diffusion is simple: you start with a clear image, progressively add noise until it's unrecognizable, and then you train a neural network to reverse that process step-by-step. To generate a new image, you just give the trained model random noise and let it work its magic.  
> *Why go through this whole process?*  
> The idea is that we want to sample data points from our distribution. Now generating data points from scratch is hard, but generating noise (random numbers) is easy, so if we learn a way to turn noise into data, we have effectively generated data samples from our distribution.

After 400 epochs, my model was generating clear digits from every class.

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; align-items: end;">
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/training-gif.gif" alt="Training Progress (GIF)" style="max-width: 80%; height: auto;">
        <br>
        <small>Training Progress (GIF)</small>
    </div>
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/training.png" alt="Final Training Output" style="max-width: 100%; height: auto;">
        <br>
        <small>Final Training Output</small>
    </div>
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/naive-deletion.png" alt="Naive Deletion" style="max-width: 100%; height: auto;">
        <br>
        <small>Naive Deletion. Finetuned for 100 epochs, increasing from left to right, sampled every 10 epochs. </small>
    </div>
</div>

Now for the hard part: making it forget.

### Attempt #1: The Naive Deletion Method

My first idea was the most straightforward one. What if I just took my fully trained model and fine-tuned it for a few more epochs on a new dataset where I'd removed all the images of the number '9'? 

I finetuned the model for 100 epochs on a new dataset containing 0 examples of the number '9'. We can see that the model does not unlearn '9', which is somewhat to be expected.

---

## Attempt #2: Using Erased Stable Diffusion (ESD)

I turned to existing research in the field, drawing inspiration from a technique called **Erased Stable Diffusion (ESD)**, which was originally designed for continuous models and adapted it for discrete diffusion models.

The idea behind ESD is to guide the model away from the concept you want to erase. Instead of just showing it examples of what to generate, you actively teach it what *not* to generate. This is done by modifying the training loss. At each step, you calculate a "deletion target" that pushes the model's prediction away from the unwanted class.

The training objective is:

$$
\mathcal{L} \;=\; \frac{1}{N}\big\lVert \mathbf{z}_{\mathrm{finetuned,cond}} - \mathbf{t}_{\mathrm{del}} \big\rVert_2^2
$$

Where:
$$
\mathbf{t}_{\mathrm{del}} \;=\; \mathbf{z}_{\mathrm{orig,uncond}} \;-\; \alpha\big(\mathbf{z}_{\mathrm{orig,cond}} - \mathbf{z}_{\mathrm{orig,uncond}}\big)
$$

Or equivalently,
$$
\mathbf{t}_{\mathrm{del}} \;=\; (1+\alpha)\,\mathbf{z}_{\mathrm{orig,uncond}} \;-\; \alpha\,\mathbf{z}_{\mathrm{orig,cond}}
$$


Here, $\mathbf{z}$ represents the logits either conditioned on the concept we want to erase,
$\mathbf{z}_{\mathrm{orig,cond}}$ or not conditioned $\mathbf{z}_{\mathrm{orig,uncond}}$, and 
$\alpha$ (or superfactor) is a scalar that controls how strongly we want to erase the concept.

My first experiment with this new loss function involved fine-tuning the *entire model*. The results were... not great. 

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/esd-full_model.png" alt="ESD Deletion; Superfactor = 1" style="max-width: 100%; height: auto;">
        <br>
        <small>Superfactor = 1</small>
    </div>
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/esd-full-superfactor2.png" alt="ESD Deletion; Superfactor = 2" style="max-width: 100%; height: auto;">
        <br>
        <small>Superfactor = 2</small>
    </div>
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/esd-full-superfactor5.png" alt="ESD Deletion; Superfactor = 5" style="max-width: 100%; height: auto;">
        <br>
        <small>Superfactor = 5</small>
    </div>
</div>

> Fine-tuning the entire model with the ESD loss is a technique that is too extreme, resulting in model collapse. A more targeted approach would be needed.

---

## Attempt #3: Using Erased Stable Diffusion (ESD) on only cross attention layers

The issue wasn't the loss function itself, but *what* I was applying it to. Fine-tuning all the model's weights was too disruptive. I needed to find the specific parts of the model that were most responsible for linking the *concept* of '9' to the visual *representation* of '9'.

That led me to the **cross-attention layers**. In a conditional diffusion model like this one, the cross-attention mechanism is the bridge between the condition (in this case, the label of the digit to generate) and the image being generated.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/cross-attn-explainer.png" alt="Cross Attention layers" style="max-width: 100%; height: auto;">
        <br>
        <small>Cross Attention layers</small>
    </div>
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/esd-crossattn.png" alt="ESD Cross Attention" style="max-width: 100%; height: auto;">
        <br>
        <small>ESD Cross Attention</small>
    </div>
</div>

<div style="display: grid; grid-template-columns: 1fr; gap: 16px;">
    <div style="text-align: center;">
        <img src="/assets/img/discrete_diffusion/esd-crossattn-100epochs.png" alt="Cross Attention layers; 100 epochs" style="max-width: 60%; height: auto;">
        <br>
        <small>Cross Attention layers; 100 epochs</small>
    </div>
</div>

> This worked much better. By fine-tuning only the cross-attention layers, the model stopped generating the target class while the quality of the other classes remained high.

By focusing the ESD loss solely on these layers, I was able to effectively sever the connection between the label '9' and its learned representation. This worked remarkably well, even with only a fraction of the original training data used for the deletion process.

---

### Reflection

This project was a great learning experience.

* **The Math is Hard...** I spent a lot of time buried in the complex mathematics of diffusion models. While understanding the theory is important, i believe and a note to my future self, is to not be afraid to "give up on the math earlier" and start experimenting when you hit a theoretical wall and learn as you go.
* **It might take a while** From the initial poor results to the long hours spent waiting for models to fine-tune on my own GPU, this project had its share of challenges. Documentation! tracking which runs are what and having a systematic way to train and evaluate is really important.

---

### Conclusion and Next Steps

My key finding is that for discrete diffusion models, selectively fine-tuning the cross-attention layers is a powerful and efficient technique for concept erasure. It provides a pathway for modifying model behavior in a targeted way, which has significant implications for AI safety—from preventing the generation of harmful content to removing copyrighted data.

But this is just the beginning. I'm excited to extend this work to a more complex, real-world domain. My next step is to train a discrete diffusion model to generate novel enzyme sequences and then use these unlearning techniques to erase specific functional classes.

Thank you for reading!
---

**Resources:**
* **Primary GitHub Repository Used:** [https://github.com/cloneofsimo/d3pm/tree/main](https://github.com/cloneofsimo/d3pm/tree/main) 
* **Relevant Paper:** [Erasing Concepts from Diffusion Models](https://arxiv.org/pdf/2303.07345) 