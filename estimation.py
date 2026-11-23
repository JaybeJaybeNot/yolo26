import armor_pnp as ap

ap.configure(width=1280, height=720, hfov=70)     # or ap.configure(K=K, dist=dist)

'''boxes, kpts = detect(model, frame)                # boxes [N,4] xyxy, kpts [N,4,2]
out = ap.PnP(boxes, kpts)                         # all plates, one call
out.valid, out.t, out.R, out.err_px  '''