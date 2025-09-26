# 🐑 Machine Learning-assisted Self-powered Ear Tag for Animal Welfare

---

## **Background & Motivation**
In modern livestock production, **animal welfare** has become a critical factor influencing livestock health, productivity, and sustainability, and is facing escalating challenges.  

Among them, disruptions in **metabolic homeostasis**—particularly **electrolyte imbalance** caused by improper dietary ratios—can lead to **chronic physiological stress, impaired immune function, and prolonged suffering** of livestock.  

These **subclinical conditions** often persist silently for days or even weeks before overt symptoms emerge. Unfortunately, current welfare assessment methods rely heavily on **subjective observation** or **intermittent, invasive blood sampling**, resulting in delayed diagnosis and missed opportunities for early intervention.  

👉 The latent nature of these injuries often leads to **irreversible harm or even mortality**.  
Therefore, there is an **urgent need for dynamic, real-time, and sustainable diagnostic tools** to monitor metabolic health and endocrine responses in livestock.  

---

## **Introduction**
To address this urgent need, we have developed a **machine learning-assisted self-powered ear tag (iSPET)** — a highly integrated and lightweight wearable platform that combines:  
- **Hybrid energy harvesting**  
- **Multiplexed biosensing**  
- **ML-driven metabolic assessment**  

Considering the complex behavior of livestock and rearing environments:  
- A **hybrid self-powered module** integrates **triboelectric nanogenerator (TENG)** and **solar cell** for efficient energy harvesting.  
  - **TENG**: captures kinetic energy of livestock to compensate for output decline under low light.  
  - **Solar cell**: ensures continuous energy supply when light intensity is stable.  

A **biocompatible microneedle array** enables **continuous monitoring** of **pH, K⁺, and Ca²⁺** levels in the interstitial fluid of livestock.  

With the assistance of a **machine learning model**, the metabolic states (**normal, acidosis, alkalosis, hypocalcification**) are **accurately distinguished**, achieving a nearly perfect **average accuracy of 98.95%**.  

---

## **How to Use**
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/wispcarey/ear-tag-ml-welfare.git

How to Use
1. Clone the Repository: git clone https://github.com/wispcarey/AutoKG.git
2. Prerequisites: Python 3.9 or 3.10
3. Installation: Install necessary packages:pip install -r requirements.txt
4. Your OpenAI API key: Input your openai api key in config
