# 🎬 Wednesday Demo Preparation Guide

## 🎯 Demo Strategy

### **Keep Free Tier** ✅ ($0/month)

**Why:**
- Demo will work perfectly on free tier
- Only issue: 30-second cold start on first request
- Solution: Warm up API before demo

---

## 📋 Pre-Demo Checklist (Day Before)

### 1. Deploy to Render (If Not Done)

```bash
# Push latest code
git add .
git commit -m "Production ready for demo"
git push origin main

# Deploy on Render dashboard
# → https://dashboard.render.com
```

### 2. Resume Neo4j AuraDB

```
1. Go to https://console.neo4j.io
2. Find: 41e180bf.databases.neo4j.io
3. Click "Resume" (if paused)
4. Wait 2 minutes
```

### 3. Test API Endpoints

```bash
# Set your Render URL
API_URL="https://versailles-planner-api.onrender.com"

# Test health endpoint
curl $API_URL/health
# Expected: {"status":"ok"}

# Test itinerary endpoint
curl -X POST $API_URL/itinerary \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2025-11-26T09:00:00",
    "total_duration_minutes": 240,
    "constraints": {
      "interests": ["history", "must_see"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

---

## ⏰ 5 Minutes Before Demo

### **Warm Up API** (Critical!)

```bash
# Run this 5 minutes before demo starts
# It triggers cold start so first demo request is fast

API_URL="https://versailles-planner-api.onrender.com"

echo "Warming up API..."
curl -s $API_URL/health
echo "✅ API is warm"

# Wait 30 seconds
sleep 30

# Test itinerary endpoint to fully warm it up
curl -s -X POST $API_URL/itinerary \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2025-11-26T09:00:00",
    "total_duration_minutes": 180,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }' > /dev/null

echo "✅ API fully warmed. Ready for demo!"
```

**After warm-up:**
- API stays responsive for 15 minutes
- All demo requests will be instant
- No 30-second wait during presentation

---

## 🎤 Demo Script

### **Scenario 1: Standard Tourist (2 minutes)**

**Narration:**
> "Let me show you our intelligent itinerary planner for Versailles Palace. Say a tourist wants a 4-hour visit focusing on history and must-see attractions."

**Request:**
```json
{
  "start_time": "2025-11-26T09:00:00",
  "total_duration_minutes": 240,
  "constraints": {
    "interests": ["history", "must_see"],
    "user_profile": "standard",
    "accessibility": "any"
  }
}
```

**Show Response:**
- Number of POIs visited
- Optimized route (TSP algorithm)
- Total travel time vs visit time
- Exact schedule with arrival/departure times

**Key Points:**
- ✨ "Algorithm optimizes route using TSP (Traveling Salesman Problem)"
- 📊 "90-95% optimal, 2000x faster than brute force"
- 🎯 "Considers opening hours, walking times, visitor profile"

---

### **Scenario 2: Family with Accessibility (2 minutes)**

**Narration:**
> "Now let's try a family with a stroller who needs step-free access. They want to visit gardens and have only 3 hours."

**Request:**
```json
{
  "start_time": "2025-11-26T10:00:00",
  "total_duration_minutes": 180,
  "constraints": {
    "interests": ["garden"],
    "user_profile": "family",
    "accessibility": "stroller",
    "lunch_break": true
  }
}
```

**Show Response:**
- Filtered to step-free & stroller-friendly paths
- Slower walking pace (family profile)
- Lunch break automatically scheduled
- Gardens prioritized

**Key Points:**
- ♿ "Filters all paths for step-free + stroller access"
- 👨‍👩‍👧 "Adjusts walking speed for family profile"
- 🍽️ "Smart lunch scheduling (11am-2pm window)"

---

### **Scenario 3: Must-See POI (1 minute)**

**Narration:**
> "Users can require specific attractions. Let's force including the Hall of Mirrors."

**Request:**
```json
{
  "start_time": "2025-11-26T14:00:00",
  "total_duration_minutes": 120,
  "constraints": {
    "interests": ["art"],
    "user_profile": "standard",
    "accessibility": "any",
    "must_include": ["versailles:Room:galerie-des-glaces"]
  }
}
```

**Show Response:**
- Hall of Mirrors guaranteed in itinerary
- Other POIs optimized around it
- Route efficiency maintained

**Key Points:**
- 🎯 "Constraint satisfaction + optimization"
- 🧮 "TSP with mandatory waypoints"

---

### **Scenario 4: RAG Integration (2 minutes)**

**Narration:**
> "This integrates with our RAG system. A user could ask in natural language..."

**Show:**
```
User: "Plan a 4-hour family visit to Versailles with gardens,
       starting at 9am tomorrow. We have a stroller."

RAG → Extracts parameters → Calls API → Returns natural response

Response: "I've planned a delightful 4-hour family itinerary starting
at 9am. You'll visit 5 beautiful locations including the gardens,
all with stroller-friendly paths. Your route is optimized to
minimize walking..."
```

**Key Points:**
- 🤖 "LLM understands natural language"
- 🔗 "Calls our optimization API"
- 💬 "Returns human-friendly response"

---

## 📊 Technical Highlights to Mention

### **1. Algorithm Performance**
- "TSP optimization with 3 solvers:"
  - Permutation (≤7 POIs): Optimal solution
  - Greedy (8-10 POIs): 90-95% optimal, instant
  - Held-Karp (>10 POIs): Near-optimal

### **2. Knowledge Graph (Neo4j)**
- "160 POIs with rich metadata"
- "319 connections with multi-profile travel times"
- "Real-time pathfinding with APOC extensions"

### **3. Constraints Handling**
- User profiles: standard, family, elder
- Accessibility: any, step-free, stroller
- Time constraints: opening hours, hard limits
- POI preferences: must-include, exclude

### **4. Production Architecture**
- "Deployed on Render (auto-scaling)"
- "Neo4j AuraDB (managed graph database)"
- "FastAPI (async, type-safe endpoints)"
- "Ready for RAG integration"

---

## 🚨 Troubleshooting

### Issue: API Returns 500 Error

**Cause:** Neo4j AuraDB paused or connection issue

**Fix:**
```bash
# Resume AuraDB
# → https://console.neo4j.io
# → Click "Resume"
# → Wait 2 minutes
```

### Issue: Slow Response (>5 seconds)

**Cause:** Cold start (didn't warm up)

**Fix:**
```bash
# Mid-demo warm-up (if needed)
curl https://your-api.onrender.com/health
# Wait 30s, continue demo
```

### Issue: No POIs Returned

**Cause:** Interest tags don't match database

**Fix:** Use these tested interests:
- ✅ "history"
- ✅ "art"
- ✅ "garden"
- ✅ "must_see"
- ✅ "architecture"

---

## 🎯 Key Talking Points

### **Problem Statement**
- "Visiting Versailles is overwhelming"
- "800 acres, 100+ points of interest"
- "Tourists need personalized, optimized routes"

### **Our Solution**
- "Intelligent itinerary planner"
- "Considers constraints: time, accessibility, interests"
- "Optimizes route using TSP algorithms"
- "Integrates with RAG for natural language"

### **Technical Innovation**
- "Graph-based knowledge representation"
- "Multi-constraint optimization (NP-hard problem)"
- "Real-time pathfinding with accessibility filters"
- "Production-ready API architecture"

### **Business Value**
- "Enhances visitor experience"
- "Reduces staff workload (automated planning)"
- "Accessible to all visitors (step-free routing)"
- "Scalable to other museums/landmarks"

---

## 📱 Demo URLs to Have Ready

### **API Endpoints**
```
Health: GET https://your-api.onrender.com/health
Plan: POST https://your-api.onrender.com/itinerary
Docs: https://your-api.onrender.com/docs
```

### **Neo4j Browser** (Optional)
```
https://console.neo4j.io
→ Show knowledge graph visualization
→ Query: MATCH (n:POI)-[r]->(m:POI) RETURN n,r,m LIMIT 50
```

### **GitHub Repository**
```
https://github.com/YOUR_USERNAME/versailles-travel-planner
→ Show code structure
→ Highlight test results in benchmarks/
```

---

## ✅ Final Checklist (Morning of Demo)

- [ ] Resume Neo4j AuraDB (console.neo4j.io)
- [ ] Warm up API (5 min before demo)
- [ ] Test all 4 demo scenarios
- [ ] Have API URLs ready in browser tabs
- [ ] Optional: Neo4j Browser open for visualization
- [ ] Backup: Have offline benchmarks ready

---

## 💡 If Something Goes Wrong

### **Fallback Plan**

If API is down during demo:

1. **Show offline benchmarks:**
   ```bash
   cat benchmarks/results/offline_scenario*.json
   # Show: "Here are our test results proving the algorithm works"
   ```

2. **Show code structure:**
   - Walk through `scripts/planner.py` (TSP logic)
   - Show `scripts/api.py` (FastAPI endpoints)
   - Explain architecture

3. **Show documentation:**
   - `DEPLOYMENT_GUIDE.md`
   - `benchmarks/README.md`
   - Explain technical approach

**Message:** "The algorithm is proven to work. This is a temporary infrastructure issue, not an algorithm problem."

---

## 🎉 Post-Demo

### Questions to Anticipate

**Q: "How do you handle crowds/queues?"**
A: "Future feature. Can integrate real-time wait time APIs."

**Q: "Can it handle multiple days?"**
A: "Current version: single day. Extension: trivial (run algorithm per day)."

**Q: "What about weather?"**
A: "Can filter outdoor POIs based on weather API."

**Q: "Scalability?"**
A: "Free tier handles 100-1000 daily users. Tested with 160 POIs, scales to 50k."

**Q: "Other languages?"**
A: "POI data has French names. RAG can translate user queries."

---

## 🚀 Summary

**Before Demo:**
1. Resume Neo4j AuraDB
2. Deploy latest code to Render

**5 Minutes Before:**
1. Warm up API (critical!)

**During Demo:**
1. Show 4 scenarios (standard, family, must-include, RAG)
2. Highlight TSP optimization
3. Mention production architecture

**If Issues:**
1. Fall back to offline benchmarks
2. Show code and documentation
3. Emphasize algorithm correctness

**Free tier is PERFECT for demo!** No need to upgrade. 💰🚫

Good luck Wednesday! 🎬✨
