Added AI-friendly documentation features to improve interaction with Large Language Models and AI coding assistants:

- Added ``llms.txt`` file following the llms.txt specification to provide structured project information for AI systems
- Added ``generate_ai_context()`` function to ``hyperspy.api`` for easy generation of AI context files from Python
- Integrated AI assistance guidance into User and Developer Guides with simplified Python-based workflow
- AI systems can now better understand HyperSpy's architecture, navigation/signal dimensions concept, extension ecosystem, and common workflows

The new ``generate_ai_context()`` function provides a cross-platform solution that eliminates the need for complex command-line instructions, automatically locates the local ``llms.txt`` file, and optionally includes web content for comprehensive context generation.
