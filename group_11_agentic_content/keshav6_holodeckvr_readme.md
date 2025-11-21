# HOLODECK VR - AI-Powered Scene Generation for Meta Quest 2

An interactive VR system that generates 3D environments from natural language descriptions using local LLMs. Type "bedroom" or "office" in VR, and watch as a fully furnished room materializes around you with properly positioned objects.

**Inspired by:** [HOLODECK: Language Guided Generation of 3D Embodied AI Environments](https://yueyang1996.github.io/holodeck/) (Yang et al., 2023)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

HOLODECK VR is a research prototype that demonstrates automated 3D scene generation in virtual reality. Unlike the original HOLODECK paper which uses constraint-based solvers, our implementation uses direct coordinate generation through prompt engineering with a local LLM (Llama 3.1 via Ollama).

**Key Innovation:** Real-time VR scene generation with zero API costs using locally-hosted AI.

### Demo Video
![INPUT](https://github.com/user-attachments/assets/68d85fde-1f3b-4ebe-94cb-08d41016d614)


---

## ✨ Features

- ✅ **Text-to-3D Scene Generation** - Natural language to fully furnished environments
- ✅ **VR-Native Interface** - Canvas-based input using Quest 2 hand controllers
- ✅ **Local AI Processing** - Runs Llama 3.1 via Ollama (no cloud APIs)
- ✅ **Smart Object Placement** - Automatic spacing with 1.5-3m clearances
- ✅ **Multiple Room Types** - Bedrooms, offices, kitchens, bathrooms, etc.
- ✅ **Real-Time Generation** - Scenes spawn in <10 seconds
- ✅ **Free Assets** - Uses Kenney.nl 3D models (CC0 license)

---

## 🏗️ Architecture

```
User Input (VR Canvas)
    ↓
DirectOllamaClient.cs → Ollama API (Local)
    ↓
JSON Response (objects, positions, descriptions)
    ↓
AssetManager.cs → Maps names to Unity prefabs
    ↓
SceneSpawner.cs → Instantiates objects in 3D space
    ↓
Generated Scene in VR
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **DirectOllamaClient.cs** | Communicates with Ollama API, manages prompts |
| **AssetManager.cs** | Maps object names to Unity prefabs |
| **SceneSpawner.cs** | Instantiates and positions 3D objects |
| **XR Origin** | VR camera and hand controller integration |

---

## 📦 Requirements

### Hardware
- **Meta Quest 2** (or Quest 3/Pro)
- **Development PC** with:
  - Windows 10/11 or macOS
  - 16GB+ RAM recommended
  - USB-C cable for Quest 2 connection

### Software
- **Unity 2022.3 LTS** or later
- **Ollama** (local LLM server)
- **Android Build Support** for Unity
- **Meta XR All-in-One SDK**

### Assets
- Kenney.nl 3D assets (included in project)

---

## 🚀 Installation

### Step 1: Download the Unity Project

1. Download the project ZIP file from Box:
   - See `keshav6_holodeckvr.txt` in this repository for the Box link
   - File size: ~2GB compressed (includes all assets)

2. Extract the ZIP file to your desired location:
   ```
   holodeck-vr/
   ├── Assets/
   ├── Packages/
   ├── ProjectSettings/
   └── ...
   ```

### Step 2: Install Ollama

1. **Download Ollama:**
   - Visit: https://ollama.com/download
   - Install for your operating system (Windows/Mac/Linux)

2. **Download Llama 3.1 model:**
   ```bash
   ollama pull llama3.1:8b
   ```

3. **Verify Ollama is running:**
   ```bash
   ollama list
   ```
   You should see `llama3.1:8b` in the list.

4. **Start Ollama server** (if not auto-started):
   ```bash
   ollama serve
   ```
   The server runs on `http://localhost:11434` by default.

### Step 3: Install Unity and Dependencies

1. **Install Unity Hub:**
   - Download from: https://unity.com/download

2. **Install Unity 2022.3 LTS:**
   - Open Unity Hub → Installs → Add
   - Select Unity 2022.3 LTS
   - Include modules:
     - ✅ Android Build Support
     - ✅ Android SDK & NDK Tools
     - ✅ OpenJDK

3. **Add the project to Unity Hub:**
   - Unity Hub → Projects → Add
   - Select the extracted `holodeck-vr` folder
   - Open the project (first load may take 5-10 minutes)

---

## ⚙️ Setup Instructions

### Configure Quest 2 for Development

1. **Enable Developer Mode:**
   - Install Meta Quest mobile app on your phone
   - Go to Menu → Devices → Your Quest 2 → Settings
   - Enable Developer Mode (requires developer account)

2. **Connect Quest 2 to PC:**
   - Use USB-C cable
   - Put on headset and allow USB debugging when prompted

3. **Verify connection:**
   ```bash
   adb devices
   ```
   You should see your Quest 2 listed.

### Configure Unity Project

1. **Open the project in Unity 2022.3 LTS**

2. **Verify XR Plugin Management:**
   - Edit → Project Settings → XR Plugin Management
   - Switch to Android tab
   - Ensure "Oculus" is checked ✅

3. **Configure Ollama URL:**
   - Open scene: `Assets/Scenes/MainScene.unity`
   - Find `LLM_MANAGER` GameObject in hierarchy
   - Select DirectOllamaClient component
   - Set **Ollama URL** to: `http://localhost:11434/api/generate`
   - Set **Model** to: `llama3.1:8b`

4. **Wire up scene references:**
   - **LLM_MANAGER → DirectOllamaClient:**
     - Prompt Input: Drag `Canvas/PromptInput` (Input Field)
     - Response Text: Drag `Canvas/ResponseText` (Text)
     - Send Button: Drag `Canvas/GenerateButton` (Button)
     - Scene Spawner: Drag `SceneManager` GameObject
   
   - **SceneManager → AssetManager:**
     - Verify prefab mappings (should be pre-configured)
   
   - **SceneManager → SceneSpawner:**
     - Asset Manager: Drag `SceneManager` (same object)
     - Spawn Parent: Drag `SpawnedObjects` GameObject

### Build Settings

1. **Switch to Android platform:**
   - File → Build Settings
   - Select Android
   - Click "Switch Platform" (may take several minutes)

2. **Configure Player Settings:**
   - Click "Player Settings" button
   - **Other Settings:**
     - Minimum API Level: Android 10.0 (API 29) or higher
     - Scripting Backend: IL2CPP
     - Target Architectures: ARM64 only (uncheck ARMv7)
   - **XR Settings:**
     - Ensure Oculus is enabled

---

## 🎮 Usage

### Running in Unity Editor (Testing)

**Note:** Full VR features require Quest 2. In editor, you can test the LLM/spawning without VR.

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Press Play in Unity editor

3. Type a room description in the UI (e.g., "bedroom")

4. Click Generate button

5. Objects should spawn in the scene view

### Deploying to Quest 2

1. **Connect Quest 2 via USB-C**

2. **Build and Run:**
   - File → Build Settings
   - Ensure scene `Assets/Scenes/MainScene.unity` is checked
   - Click "Build And Run"
   - Choose a save location for APK
   - Wait for build and deployment (5-10 minutes first time)

3. **Using in VR:**
   - Put on Quest 2 headset
   - App launches automatically after deployment
   - Use hand controllers to interact with canvas
   - Type room description (e.g., "office", "bedroom", "kitchen")
   - Press Generate button
   - Watch as objects spawn around you!

### Example Prompts

Try these prompts for best results:

- ✅ **Simple:** "bedroom", "office", "kitchen"
- ✅ **Descriptive:** "modern office", "cozy bedroom", "spacious living room"
- ✅ **Specific:** "office with desk and couch", "bedroom with large bed"
- ❌ **Too complex:** "futuristic sci-fi spaceship cockpit with holographic displays" (limited asset library)

---

## 📁 Project Structure

```
holodeck-vr/
├── Assets/
│   ├── Scenes/
│   │   └── SampleScene.unity          # Main VR scene
│   ├── Scripts/
│   │   ├── DirectOllamaClient.cs    # LLM communication
│   │   ├── AssetManager.cs          # Prefab mapping
│   │   └── SceneSpawner.cs          # Object instantiation
│   ├── Prefabs/
│   │   
│   │   
│   │   
│   └── Materials/                   # Textures and materials
├── Packages/
│   └── manifest.json                # Package dependencies
├── ProjectSettings/
└── README.md
```

### Key Scripts Explained

#### `DirectOllamaClient.cs`
- Sends HTTP POST requests to Ollama API
- Contains system prompt with spacing rules
- Parses JSON responses into SceneData objects
- Handles error cases and retries

#### `AssetManager.cs`
- Maintains Dictionary mapping object names → prefabs
- Supports fuzzy matching (e.g., "desk" matches "office_desk")
- Provides GetPrefab() method for spawner

#### `SceneSpawner.cs`
- Instantiates prefabs at specified positions
- Implements fallback grid layout if objects are clustered
- Handles scene clearing and object management

---

## 🔧 Technical Details

### Object Positioning Algorithm

We use **direct coordinate generation** rather than HOLODECK's constraint-based solver:

1. **Prompt Engineering:** LLM receives explicit spacing rules:
   - Beds: 2-3m clearance
   - Tables/chairs: 1.5-2m separation
   - Bookcases: against walls (x or z at ±4 to ±5)
   - Small items: ≥1.5m from furniture

2. **Example-Based Learning:** Prompt includes JSON example with proper spacing:
   ```json
   {"name": "bed", "x": -2, "z": -3},
   {"name": "table", "x": 3, "z": -3}  // 5 meters apart
   ```

3. **Coordinate System:**
   - Origin (0, 0, 0) at room center
   - X-axis: left(-) to right(+)
   - Y-axis: floor(0) to ceiling(+)
   - Z-axis: back(-) to front(+)
   - Room bounds: typically -5 to +5 meters

4. **Fallback System:** If LLM generates clustered objects (all within 2m), automatic grid redistribution kicks in

### Comparison to HOLODECK Paper

| Feature | HOLODECK (Paper) | Our Implementation |
|---------|------------------|-------------------|
| **Approach** | Constraint-based solver | Direct coordinate generation |
| **LLM** | GPT-4 | Llama 3.1 (local) |
| **Asset Library** | Objaverse (50K+) | Kenney.nl (~100) |
| **Use Case** | Offline training data | Real-time VR interaction |
| **Scale** | Large-scale batch | Single-room interactive |
| **Cost** | API calls required | Zero (local) |

---

## 🐛 Troubleshooting

### Ollama Connection Issues

**Problem:** "Failed to connect to Ollama"

**Solutions:**
1. Verify Ollama is running:
   ```bash
   ollama list
   ollama serve
   ```

2. Check URL in Unity:
   - Should be `http://localhost:11434/api/generate`
   - If running on different machine, update IP address

3. Test Ollama directly:
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "llama3.1:8b",
     "prompt": "Hello"
   }'
   ```

### Quest 2 Deployment Issues

**Problem:** App doesn't deploy to Quest 2

**Solutions:**
1. Check USB connection and ADB:
   ```bash
   adb devices
   ```

2. Ensure Developer Mode is enabled in Quest settings

3. Allow USB debugging on headset (prompt appears when connecting)

4. Try different USB cable/port

### Objects Spawn Clustered

**Problem:** All objects spawn in one small area

**Solutions:**
1. This means LLM didn't follow spacing rules
2. Check system prompt in DirectOllamaClient.cs
3. Try regenerating scene (click Generate again)
4. Fallback grid system should activate automatically

### Assets Missing

**Problem:** Objects spawn as pink cubes or don't appear

**Solutions:**
1. Verify asset library downloaded completely
2. Check AssetManager prefab mappings in Inspector
3. Re-import Kenney.nl assets if needed

### Performance Issues in VR

**Problem:** Low framerate, stuttering

**Solutions:**
1. Reduce number of objects spawned (modify prompt)
2. Lower texture quality in Quality Settings
3. Disable shadows on spawned objects
4. Use simpler 3D models

---

## 📚 References

### Research Papers
- [HOLODECK: Language Guided Generation of 3D Embodied AI Environments](https://yueyang1996.github.io/holodeck/)
  - Yang, Y., Jiang, H., Mei, S., & Yao, L. (2023)

### Tools & Libraries
- [Ollama](https://ollama.com/) - Local LLM runtime
- [Unity XR Plugin Management](https://docs.unity3d.com/Packages/com.unity.xr.management@4.0/manual/index.html)
- [Meta XR SDK](https://developer.oculus.com/documentation/unity/)

### Assets
- [Kenney.nl Asset Library](https://kenney.nl/assets) - CC0 Licensed 3D models
  - Furniture Kit
  - Office Kit
  - Kitchen Kit

---

## 🙏 Acknowledgments

- **Yue Yang et al.** for the HOLODECK research paper that inspired this project
- **Kenney.nl** for providing high-quality free 3D assets
- **Meta** for Quest 2 and XR SDK documentation
- **Ollama team** for making local LLM deployment accessible

---

### Future Improvements
- [ ] Voice command input instead of typing
- [ ] Multi-room layouts with connected spaces
- [ ] Wall/floor generation
- [ ] Object interaction (grabbing, moving)
- [ ] Save/load generated scenes
- [ ] Larger asset library integration
- [ ] Style variations (modern, medieval, sci-fi)
- [ ] Lighting system generation

---

**Built with ❤️ for VR and AI research**
