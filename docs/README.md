# NeuroBridge Documentation

**Last Updated:** 2025-10-06 (v0.1.9)

This directory contains detailed technical documentation for the NeuroBridge library.

---

## 📚 Available Documentation

### **API roadmap.md** - Implementation Status Tracker
**Status:** ✅ Updated (2025-10-06)

Tracks the implementation status of the connection API features. Originally a planning document, now serves as a reference for what's implemented, what's not, and why.

**Contents:**
- ✅ Core connection operator (COMPLETED)
- ✅ Function-based parameters (COMPLETED)
- ✅ Spatial connectivity (COMPLETED)
- ✅ Filtering and masking (COMPLETED)
- ❌ Network container class (NOT IMPLEMENTED - by design)
- ❌ Connection templates (NOT IMPLEMENTED - low priority)

**Use this when:** You want to know if a feature exists and where to find it in the code.

---

### **API specs.md** - Current API Reference
**Status:** ✅ Updated (2025-10-06)

Complete specification of the working v0.1.9 API. This is the **main user reference document**.

**Contents:**
- Connection operator `>>` with all parameters
- Connection patterns (all-to-all, random, distance, one-to-one)
- Function-based weight/delay specification
- Neuron models (ParrotNeurons, SimpleIFNeurons, IFNeurons, RandomSpikeNeurons)
- Monitoring (SpikeMonitor, VariableMonitor, RingBufferSpikeMonitor)
- Experiment framework
- Multi-GPU setup
- Performance tips

**Use this when:** You're implementing something and need to know the exact API syntax and parameters.

---

### **Architectural overview.md** - Conceptual Design Guide
**Status:** ✅ Updated (2025-10-06)

In-depth explanation of NeuroBridge's internal architecture and design philosophy.

**Contents:**
- Purpose, audience, and niche
- Node hierarchy and execution model
- Simulator and LocalCircuit architecture
- Groups and filtering system
- Neuron models and synaptic connections
- Inter-GPU communication (BridgeNeuronGroup)
- Monitors and data extraction
- Experiment lifecycle
- Performance considerations
- Extensibility guidelines

**Use this when:** You want to understand *why* the library is designed this way, or you're extending/modifying the core.

---

## 🧭 Quick Navigation Guide

### For New Users:
1. Start with **root `/README.md`** (if it exists) for overview and installation
2. Read **`API specs.md`** sections 1-3 for connection basics
3. Try examples in `/examples/` directory
4. Read **`API specs.md`** sections 4-6 for advanced features

### For Development:
1. **Root `/PROJECT_STATUS.md`** - Current state, bugs, priorities
2. **Root `/ROADMAP.md`** - Long-term development plan
3. **Root `/DEVELOPMENT_PRIORITIES.md`** - What to work on next
4. **`Architectural overview.md`** - Internal design and philosophy
5. **`API roadmap.md`** - Feature implementation status

### For Understanding Code:
1. **`Architectural overview.md`** - High-level concepts
2. **`API specs.md`** - What the API does
3. **Source code docstrings** - Implementation details
4. **`/examples/`** - Working usage patterns

### For Research Use:
1. **`API specs.md`** - Everything you need to build networks
2. **Root `/PROJECT_STATUS.md`** - Known limitations and bugs
3. **`/examples/04_BRN_STDP.py`** - Complex example with STDP

---

## 📊 Documentation Status Matrix

| Document | Last Updated | Completeness | Audience |
|----------|--------------|--------------|----------|
| **API specs.md** | 2025-10-06 | ✅ Complete | Users |
| **API roadmap.md** | 2025-10-06 | ✅ Complete | Developers |
| **Architectural overview.md** | 2025-10-06 | ✅ Complete | Advanced users/Contributors |
| **Root PROJECT_STATUS.md** | 2025-10-06 | ✅ Complete | Developers |
| **Root ROADMAP.md** | 2025-10-06 | ✅ Complete | Developers |
| **Root DEVELOPMENT_PRIORITIES.md** | 2025-10-06 | ✅ Complete | Active developers |
| **Root CHANGELOG.md** | 2025-10-06 | ✅ Complete | All users |
| **Tutorials** | N/A | ❌ Not created | New users |
| **API reference (Sphinx)** | N/A | ❌ Not generated | Users |

---

## 🔮 Future Documentation Plans

### Before v0.2.0 (Public Release):
- [ ] Comprehensive getting-started tutorial
- [ ] Step-by-step Jupyter notebooks
- [ ] Auto-generated API reference (Sphinx/ReadTheDocs)
- [ ] Performance benchmarking guide
- [ ] Troubleshooting FAQ

### Long-term:
- [ ] Video tutorials
- [ ] Paper examples and reproductions
- [ ] Community contributions guide
- [ ] Advanced topics (custom neurons, exotic connectivity patterns)

---

## 💡 Tips for Reading Documentation

1. **Start practical, go deep later:** Begin with API specs and examples, dive into architecture when you need to understand internals.

2. **Documentation hierarchy:**
   ```
   Examples (try it) → API specs (syntax) → Architecture (why it works)
   ```

3. **When docs are unclear:** Check working code in `/examples/` - code doesn't lie.

4. **For bugs/issues:** Always check root `/PROJECT_STATUS.md` first - your issue might be a known limitation.

5. **Contributing improvements:** Documentation PRs are welcome! Focus on:
   - Clarifying confusing sections
   - Adding examples
   - Fixing outdated information
   - Improving code snippets

---

## 📝 Documentation Conventions

### Code Examples:
All code examples in documentation are **tested patterns** from actual working code unless marked with:
- `# Conceptual example - not tested`
- `# Future feature - not yet implemented`

### Version Specificity:
Documentation explicitly states version when features changed:
- "Since v0.1.8" = feature added/changed in that version
- "Current API" = v0.1.9 unless otherwise stated

### Parameter Defaults:
When docs show `weight=1.0`, this is the **actual default** in code, not an example value.

---

## 🔗 External Resources

- **Source Code:** `../` (parent directory - browse actual implementation)
- **Examples:** `../examples/` (working scripts you can run)
- **Tests:** `../tests/` (unit and integration tests)
- **Main README:** `../README.md` (overview and installation)
- **GitHub Issues:** (for bug reports and feature requests)
- **Paper:** (coming soon - will link to publication)

---

## ❓ Getting Help

**If you're stuck:**

1. **Check examples first:** `../examples/` - most questions answered there
2. **Search this directory:** `grep -r "your_topic" docs/`
3. **Read the architecture:** `Architectural overview.md` explains design decisions
4. **Check known issues:** Root `/PROJECT_STATUS.md` section "Known Bugs and Issues"
5. **Ask the maintainer:** (contact info in main README)

**Before filing an issue:**
- ✅ Check if documented in `/PROJECT_STATUS.md`
- ✅ Try the simplest possible reproduction case
- ✅ Include version info (`import neurobridge; print(neurobridge.__version__)`)

---

**Happy coding! 🧠⚡**
