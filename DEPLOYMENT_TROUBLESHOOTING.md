# Deployment Troubleshooting Guide

## Current Issue: OpenAI Import Error

### Problem
The deployment is failing with:
```
ModuleNotFoundError: No module named 'openai'
File "/app/wiki/wiki_processor.py", line 27, in <module>
    from openai import OpenAI
```

### Root Cause
The deployment server is using a cached version of the old `wiki/wiki_processor.py` file that still contains the OpenAI import, even though we've removed it from the codebase.

### Solutions Applied

#### ✅ Immediate Fix (Completed)
- Added `openai==1.3.0` to `requirements.txt` as a temporary fix
- This allows the deployment to work with the cached old file
- Committed and pushed in commit `d230ebf`

#### 🔄 Permanent Fix (Next Steps)

1. **Force Deployment Refresh**
   - Trigger a new deployment to clear the cache
   - In Render: Go to your service → Manual Deploy → Deploy Latest Commit
   - This should pick up the latest code without OpenAI imports

2. **Verify Deployment Branch**
   - Ensure your deployment is using the `yiftach` branch
   - Latest commit should be `d230ebf` or newer

3. **Clear Build Cache**
   - In Render: Settings → Clear Build Cache
   - Then trigger a new deployment

4. **Remove OpenAI Dependency (After Successful Deployment)**
   ```bash
   # Once deployment works with new code, remove openai from requirements.txt
   git checkout yiftach
   # Edit requirements.txt to remove openai==1.3.0 line
   git add requirements.txt
   git commit -m "🧹 Remove temporary openai dependency - deployment cache cleared"
   git push origin yiftach
   ```

### Verification Steps

1. **Check Deployment Logs**
   - Look for successful import of `wiki.wiki_processor`
   - Should not see OpenAI import errors

2. **Test API Endpoints**
   ```bash
   curl https://your-app.onrender.com/health
   curl -X POST https://your-app.onrender.com/fact-check \
     -H "Content-Type: application/json" \
     -d '{"text": "Test message", "author": "Test Author"}'
   ```

3. **Monitor Function Usage**
   - Only these functions should be used from `wiki_processor`:
     - `get_author_info()`
     - `get_author_info_by_name()`
     - `process_characters_wiki()`

### Current Codebase Status

✅ **Clean Codebase**
- No OpenAI imports in any Python files
- All unused functions removed
- Only Wikipedia-based functions remain
- All tests passing locally

✅ **Temporary Fix Applied**
- OpenAI added to requirements.txt for deployment compatibility
- Will be removed once cache is cleared

### Files Modified

1. **wiki/wiki_processor.py** (commit `b35aa00`)
   - Removed `from openai import OpenAI`
   - Removed `extract_entities_with_grok()` function
   - Removed `process_wiki_with_grok()` function
   - Kept only Wikipedia-based functions

2. **requirements.txt** (commit `d230ebf`)
   - Added `openai==1.3.0` as temporary fix
   - To be removed after deployment cache cleared

### Next Actions

1. **Immediate**: Trigger manual deployment in Render
2. **Verify**: Check deployment logs for successful startup
3. **Test**: Verify API endpoints work correctly
4. **Cleanup**: Remove openai from requirements.txt once confirmed working
5. **Monitor**: Ensure no functionality is lost

### Contact Information

If issues persist:
- Check Render deployment logs
- Verify branch and commit being deployed
- Consider rolling back to previous working version if needed
- Test locally first: `python -c "from wiki.wiki_processor import get_author_info; print('✅ Import works')"` 