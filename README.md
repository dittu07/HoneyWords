# HoneyWords
LingoShield is a multilingual honeyword-based password protection system. It generates fake but realistic passwords in Chinese, Hindi, German, and French to confuse attackers and detect breaches if a decoy is used.

# 🔐 LingoShield – Multilingual Honeyword-Based Password Protection

**LingoShield** is a security project that enhances password protection by generating *honeywords*—fake yet believable passwords that confuse attackers. By introducing uncertainty into leaked password datasets, LingoShield makes it significantly harder for attackers to identify the real password, even if they gain access to the system.

## 🌍 Supported Languages
- 🇨🇳 Chinese
- 🇮🇳 Hindi
- 🇩🇪 German
- 🇫🇷 French

Each language has tailored logic to ensure that decoy passwords are linguistically realistic and culturally relevant.

---

## ⚙️ How It Works

1. **Password Input:** A real user password is provided.
2. **Decoy Generation:** Several fake passwords are generated using language-specific models or rules.
3. **Storage:** The real password and decoys are stored together in a shuffled list.
4. **Login Monitoring:** If a honeyword is ever used in a login attempt, an alert is triggered — indicating a likely breach.

This method is based on the concept of *honeywords* introduced in [Juels and Rivest, 2013](https://people.csail.mit.edu/rivest/pubs/JR13.pdf), extended with multilingual capabilities.

---

## 🎯 Why Use LingoShield?

- 🧠 **Harder to Guess:** Decoys closely resemble the real password.
- 🛡️ **Breach Detection:** Triggers alerts if a honeyword is used.
- 🌐 **Multilingual Flexibility:** Protects users across diverse linguistic backgrounds.
- 🔄 **Seamless Integration:** Can be integrated into existing authentication systems.

---

## 📦 Installation

```bash
git clone https://github.com/dittu07/lingoshield.git
cd lingoshield
pip install -r requirements.txt

