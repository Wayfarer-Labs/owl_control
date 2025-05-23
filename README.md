# OWL Control

**Help train the next generation of AI by sharing your gameplay!**

OWL Control is a desktop application that records gameplay footage and input data from video games to create open-source datasets for AI research. By using OWL Control, you're contributing to the development of AI agents and world models.

## 🎮 What is OWL Control?

OWL Control automatically records your gameplay sessions (video + keyboard/mouse inputs) from supported single-player games. This data is uploaded to create a public dataset that researchers worldwide can use to train AI models.

### Key Features

- **Automatic Recording** - Detects and records supported games automatically
- **Privacy First** - Only records whitelisted single-player games, no microphone audio
- **Full Control** - Start/stop recording anytime with hotkeys (F4/F5)
- **Lightweight** - Runs quietly in your system tray

## 🚀 Getting Started

1. **Download** the latest version from the [Releases](https://github.com/OpenWorld-Labs/owl_control/releases) page
2. **Install** the application for your operating system
3. **Create an account** or enter your API key
4. **Review and accept** the data collection terms
5. **Start gaming!** OWL Control will automatically record supported games

## 🛡️ Privacy & Security

Your privacy is our top priority:

- ✅ Only records gameplay from pre-approved single-player games
- ✅ No microphone recording - your voice stays private
- ✅ No personal information captured
- ✅ All data is anonymized before public release
- ✅ You can delete recordings anytime
- ✅ Completely voluntary - stop recording whenever you want

### Content Policy

⚠️ **Important**: By using OWL Control, you agree not to upload any content that is:
- Illegal in your jurisdiction or country of origin
- Malicious, harmful, or inappropriate
- Violates any applicable laws or regulations

If you upload content that violates these terms, we will take necessary and proportional actions, which may include:
- Removal of content
- Suspension of access
- Reporting to appropriate authorities
- Other legal remedies as required

## 🎯 Supported Games

OWL Control automatically detects and records from a curated list of single-player games. The app will notify you when it detects a supported game.

## ⌨️ Controls

- **F4** - Start recording manually
- **F5** - Stop recording
- **System Tray Icon** - Access settings and controls

## 🤝 Contributing to AI Research

By using OWL Control, you're helping to:

- Train AI agents to understand and play games
- Develop better world models for AI systems
- Create open datasets for the research community
- Advance the field of AI and machine learning

All collected data will be made publicly available for research purposes.

## 💻 For Developers

OWL Control is open source! If you're interested in the technical details or want to contribute:

- Built with Electron (UI) and Python (recording backend)
- Uses OBS for high-quality video capture
- Records inputs at 60 FPS for precise action tracking

### Building from Source

```bash
# Clone the repository
git clone https://github.com/OpenWorld-Labs/owl_control.git
cd owl_control

# Install dependencies
npm install

# Run in development mode
npm run dev

# Build for production
npm run build
```

For detailed development instructions, see our [Development Guide](docs/development.md).

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋 Support

- **Issues or Bugs?** Please report them on our [GitHub Issues](https://github.com/OpenWorld-Labs/owl_control/issues) page
- **Questions?** Please visit our [GitHub Issues](https://github.com/OpenWorld-Labs/owl_control/issues) page

---

*OWL Control is a project by [Open World Labs](https://openworldlabs.com) - Building open datasets for AI research*
