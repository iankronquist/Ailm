# Ailm ᚐ

Ailm is a small Large Language Model designed to be easily trained on an M1 Max Macbook Pro.

It is named after the [letter ᚐ, pronounced Ailm](https://en.wikipedia.org/wiki/Ailm) from the Ogham alphabet which was used to write Old Irish.


I drew upon the following resources to design this model:

- HuggingFace's [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) and their accompanying [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- Andrej Karpathy's [nanoChat](https://github.com/karpathy/nanochat)
- Facebook's [Llama 2](https://arxiv.org/pdf/2307.09288)
- OpenAI's [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- PleIAs's [Baguettotron](https://huggingface.co/PleIAs/Baguettotron)

It uses Apple's MLX framework as it seems to get better performance than other frameworks I've experimented with.