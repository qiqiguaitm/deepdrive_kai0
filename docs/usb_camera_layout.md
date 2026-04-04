# USB Camera Layout Issue

## Current topology (all 3 cameras on shared hub chain)

```
Bus 002 (xhci_hcd, controller 0000:42:00.3, 10Gbps)
  ├── D405 hand_left  (409122273074)
  └── D435 top_head   (254622070889)  ← CONFLICTS, gets disconnected

Bus 004 (xhci_hcd, controller 0000:06:00.3, 10Gbps)  
  └── D405 hand_right (409122271568)
```

## Problem
When the 3rd camera starts streaming, the USB hub on Bus 002 
re-enumerates and disconnects the D435 (VIDIOC_QBUF: No such device).

## Fix
Move the D435 (top_head) to a USB port on Bus 004's controller 
(0000:06:00.3) so each controller handles at most 2 cameras:

```
Bus 002 → 1x D405
Bus 004 → 1x D405 + 1x D435
```

Or move one D405 to Bus 002 so the D435 gets a controller to itself.
