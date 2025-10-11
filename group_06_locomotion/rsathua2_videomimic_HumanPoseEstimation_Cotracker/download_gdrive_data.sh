#!/bin/bash

set -e

# === CONFIGURATION ===
FILE_ID="1ZqIRd_6PnoWviFCASXp96qa0jn6aBQQN"
TARGET_DIR="./"  # Change as needed
ZIP_NAME="downloaded_file.zip"

# === CHECK DEPENDENCIES ===
if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing gdown..."
    pip install --upgrade pip
    pip install gdown
fi

# === MAKE TARGET DIR ===
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# === DOWNLOAD ===
echo "⬇ Downloading file..."
gdown --id "$FILE_ID" --output "$ZIP_NAME"

# === UNZIP ===
echo "📦 Unzipping..."
unzip "$ZIP_NAME"

# === CLEAN UP ===
echo "🧹 Removing zip file..."
rm "$ZIP_NAME"

echo "✅ Done. Files are in $TARGET_DIR"
