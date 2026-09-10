#ifndef UNITREE_POLICY_REQUEST_H
#define UNITREE_POLICY_REQUEST_H

#include <string>

// Decide whether a policy hot-switch request names a genuinely new policy.
// Requests for the currently active path, for a path already queued, or for a
// path whose load is in flight are idempotent confirmations: they must not
// trigger a second load or another takeover reset.  Empty paths are never
// accepted.
inline bool shouldAcceptPolicyRequest(const std::string &requested,
                                      const std::string &activePath,
                                      const std::string &pendingPath,
                                      const std::string &loadingPath)
{
    if(requested.empty()){
        return false;
    }
    return requested != activePath && requested != pendingPath &&
           requested != loadingPath;
}

// Atomically (under the caller's mutex) claim the queued request: move it to
// the in-flight slot and clear the queue.  Returns the claimed path, or an
// empty string when nothing is queued.  The claimed request must not be
// re-validated against shouldAcceptPolicyRequest.
inline std::string claimPendingPolicyRequest(std::string &pendingPath,
                                             std::string &loadingPath)
{
    if(pendingPath.empty()){
        return std::string();
    }
    loadingPath = pendingPath;
    pendingPath.clear();
    return loadingPath;
}

// Finish an in-flight load.  Clears only the matching in-flight slot; on
// success promotes the claimed path to active.  Any *different* queued request
// is preserved.  Returns true when a queued request remains, so the caller can
// keep the reload flag armed.
inline bool finishPolicyLoad(const std::string &claimedPath, bool success,
                             std::string &activePath,
                             std::string &loadingPath,
                             std::string &pendingPath)
{
    if(loadingPath == claimedPath){
        loadingPath.clear();
    }
    if(success && !claimedPath.empty()){
        activePath = claimedPath;
    }
    return !pendingPath.empty();
}

#endif  // UNITREE_POLICY_REQUEST_H
