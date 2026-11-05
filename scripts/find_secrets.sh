# scripts/find_secrets.sh
#!/usr/bin/env bash
# Lists likely HF tokens in tracked files and in git history. Tokens are masked in the output.
# Fix = REVOKE the token on huggingface.co/settings/tokens, then use a Colab secret / env var.

set -u
echo "== tracked files =="

git grep -nE "hf_[A-Za-z0-9]{20,}" | sed -E 's/hf_[A-Za-z0-9]{20,}/hf_***MASKED***/'

echo "== commits whose diff adds/removes a match =="
git log --all --oneline -G"hf_[A-Za-z0-9]{20,}"