# Linstor Storage Setup for ML Workloads

## Overview

We've configured Linstor for high-performance model shard loading with the following optimizations:

1. **Zone-aware provisioning** (`WaitForFirstConsumer`) - Eliminates cross-zone latency
2. **Single replica** - Maximum write performance (trade-off: no HA)
3. **XFS filesystem** - Optimized for large sequential I/O
4. **Disabled remote volume access** - Lowest possible latency

## Files Created

- `storageclass-linstor-fast.yaml` - Optimized StorageClass for ML cache
- `pvc-cache.yaml` - Updated to use Linstor (Filesystem mode)
- `pvc-cache-block.yaml` - Optional Block mode variant (requires app changes)

## Deployment Steps

### 1. Apply the StorageClass

```bash
kubectl apply -f k8s/storageclass-linstor-fast.yaml
```

### 2. Delete existing PVC (if necessary)

**⚠️ WARNING: This will delete all cached data!**

```bash
# Check what's using the PVC
kubectl get pods -A -o json | jq -r '.items[] | select(.spec.volumes[]?.persistentVolumeClaim.claimName=="ms-swift-cache-pvc") | .metadata.name'

# If no pods are using it, delete
kubectl delete pvc ms-swift-cache-pvc

# Wait for deletion to complete
kubectl get pvc ms-swift-cache-pvc
```

### 3. Create new Linstor PVC

```bash
kubectl apply -f k8s/pvc-cache.yaml
```

### 4. Verify PVC status

```bash
# Check PVC - should be in "Pending" state (waiting for first consumer)
kubectl get pvc ms-swift-cache-pvc

# Expected output:
# NAME                  STATUS    VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS        AGE
# ms-swift-cache-pvc    Pending                                      linstor-fast-cache  5s
```

### 5. Add zone affinity to your jobs (IMPORTANT!)

To ensure optimal performance, add zone affinity to your jobs. This tells Kubernetes where to run your workload, and Linstor will create the volume in that same zone.

#### Option A: Let Kubernetes choose the best zone automatically

No changes needed! `WaitForFirstConsumer` will create the volume in whatever zone the pod lands.

#### Option B: Pin to a specific zone (recommended for predictable performance)

Add this to your job's `spec.template.spec.affinity`:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        # Pin to UCSD zone (example)
        - key: topology.kubernetes.io/zone
          operator: In
          values:
          - ucsd  # or: sdsu, gatech, chico, etc.

        # Keep existing GPU affinity
        - key: nvidia.com/gpu.product
          operator: In
          values:
          - NVIDIA-A100-SXM4-80GB
```

Available zones in your cluster:
- `ucsd`
- `sdsu`
- `gatech`
- `chico`
- `humboldt`
- `ucmerced`
- `brookdale`
- `cau`
- `onenet`
- `udel`

**Check available zones:**
```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,ZONE:.metadata.labels."topology\.kubernetes\.io/zone" | grep -v '<none>' | sort -k2
```

### 6. Launch your job

```bash
kubectl apply -f k8s/job-sft-qwen3vl-fire-4gpu.yaml
```

### 7. Monitor PVC binding

```bash
# Watch PVC status - should transition from Pending -> Bound
kubectl get pvc ms-swift-cache-pvc -w

# Check which node the pod landed on
kubectl get pods -l app=ms-swift -o wide

# Verify PV was created in the same zone
kubectl get pv -o custom-columns=NAME:.metadata.name,CAPACITY:.spec.capacity.storage,STORAGECLASS:.spec.storageClassName,NODE-AFFINITY:.spec.nodeAffinity
```

## Performance Comparison

### Before (Rook-Ceph with Immediate binding)
- ❌ PVC created immediately in random zone
- ❌ Pod might schedule in different zone
- ❌ Cross-zone network latency (1-10ms+)
- ❌ Limited bandwidth between zones

### After (Linstor with WaitForFirstConsumer)
- ✅ PVC created in same zone as pod
- ✅ Local/in-zone storage access
- ✅ Sub-millisecond latency
- ✅ Full local network bandwidth

## Benchmarking

To measure actual performance improvement:

```bash
# Create benchmark job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: storage-benchmark
spec:
  template:
    spec:
      containers:
      - name: fio
        image: dmonakhov/alpine-fio
        command:
        - fio
        - --name=seqread
        - --rw=read
        - --bs=4M
        - --size=10G
        - --numjobs=1
        - --directory=/cache
        - --output-format=json
        volumeMounts:
        - name: cache
          mountPath: /cache
      restartPolicy: Never
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: ms-swift-cache-pvc
EOF

# Check results
kubectl logs job/storage-benchmark
```

## Troubleshooting

### PVC stuck in Pending
- **Expected behavior** if no pod is using it yet
- Linstor will create the volume when a pod requests it

### Pod stuck in Pending
```bash
kubectl describe pod <pod-name>
```
Look for scheduling errors related to:
- No nodes in the requested zone
- Insufficient resources in the zone
- PVC binding issues

### Check Linstor status
```bash
# List Linstor controller pods
kubectl get pods -n piraeus-datastore

# Check Linstor resources
kubectl exec -it deploy/linstor-controller -n piraeus-datastore -- linstor resource list
kubectl exec -it deploy/linstor-controller -n piraeus-datastore -- linstor storage-pool list
```

### Performance issues
1. Verify volume is local to the node:
   ```bash
   kubectl get pv -o yaml | grep -A 10 nodeAffinity
   ```

2. Check if pod and volume are in the same zone:
   ```bash
   POD_NODE=$(kubectl get pod <pod-name> -o jsonpath='{.spec.nodeName}')
   kubectl get node $POD_NODE -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}'
   ```

## Advanced: Block Mode for Maximum Performance

If you need absolute maximum performance and can modify your application to read from a raw block device:

1. Use `pvc-cache-block.yaml` instead
2. Update your pod to use `volumeDevices` instead of `volumeMounts`:

```yaml
containers:
- name: model-loader
  volumeDevices:
  - name: cache-volume
    devicePath: /dev/cache
```

3. Access the device at `/dev/cache` in your application

**Note:** This requires application code changes to read from a block device instead of a filesystem.

## Monitoring

Add these metrics to track storage performance:

```bash
# PVC usage
kubectl get pvc ms-swift-cache-pvc -o jsonpath='{.status.capacity.storage}'

# Pod I/O metrics (if metrics-server is installed)
kubectl top pod -l app=ms-swift
```

## Rollback

If you need to switch back to Rook-Ceph:

```bash
kubectl delete pvc ms-swift-cache-pvc
# Restore old pvc-cache.yaml from git
git checkout k8s/pvc-cache.yaml
kubectl apply -f k8s/pvc-cache.yaml
```
