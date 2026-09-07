import base64
import json
from pathlib import Path
import uuid
import zlib
from IPython.display import HTML
import nibabel as nib
import numpy as np


def pqnii(
    image_path,
    overlay_path=None,
    width=400,
    height=400,
    colormap="red",
    max_dim=128,
):
    """3D Canvas NIfTI viewer using zlib compression and auto-downsampling."""
    img_path = Path(image_path).resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img = nib.load(img_path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim > 3:
        data = data[..., 0]

    # Calculate stride to auto-fit oversized volumes into safety threshold
    strides = [max(1, int(np.ceil(s / max_dim))) for s in data.shape[:3]]
    data_sub = data[:: strides[0], :: strides[1], :: strides[2]]
    shape = [int(s) for s in data_sub.shape[:3]]

    d_min, d_max = float(np.nanmin(data_sub)), float(np.nanmax(data_sub))
    norm_data = (
        ((data_sub - d_min) / (d_max - d_min) * 255)
        if d_max > d_min
        else np.zeros_like(data_sub)
    )
    uint8_data = np.clip(norm_data, 0, 255).astype(np.uint8)

    # Base64 + Zlib Compression to prevent output limit truncation
    compressed_base = zlib.compress(uint8_data.tobytes(order="C"), level=6)
    base_b64 = base64.b64encode(compressed_base).decode("ascii")

    has_overlay = overlay_path is not None
    overlay_b64 = ""
    ov_min, ov_max = 0.0, 1.0
    p5, p95 = 0.0, 1.0
    overlay_filename = ""

    if has_overlay:
        ov_path = Path(overlay_path).resolve()
        if not ov_path.exists():
            raise FileNotFoundError(f"Overlay file not found: {ov_path}")
        overlay_filename = ov_path.name
        ov_img = nib.load(ov_path)
        ov_data = np.asarray(ov_img.get_fdata(dtype=np.float32))
        if ov_data.ndim > 3:
            ov_data = ov_data[..., 0]

        ov_data_sub = ov_data[:: strides[0], :: strides[1], :: strides[2]]
        ov_min, ov_max = float(np.nanmin(ov_data_sub)), float(
            np.nanmax(ov_data_sub)
        )

        nz_mask = ov_data_sub > 0
        if np.any(nz_mask):
            p5 = float(np.percentile(ov_data_sub[nz_mask], 5))
            p95 = float(np.percentile(ov_data_sub[nz_mask], 95))
        else:
            p5, p95 = ov_min, ov_max

        ov_norm = (
            ((ov_data_sub - ov_min) / (ov_max - ov_min) * 255)
            if ov_max > ov_min
            else np.zeros_like(ov_data_sub)
        )
        uint8_ov = np.clip(ov_norm, 0, 255).astype(np.uint8)

        compressed_ov = zlib.compress(uint8_ov.tobytes(order="C"), level=6)
        overlay_b64 = base64.b64encode(compressed_ov).decode("ascii")

    uid = uuid.uuid4().hex
    canvas_id, info_id = f"cv_{uid}", f"info_{uid}"
    btn_view_id, btn_color_id = f"btn_v_{uid}", f"btn_c_{uid}"
    dropdown_view_id, dropdown_color_id = f"dd_v_{uid}", f"dd_c_{uid}"
    op_slider_id, op_val_id = f"sl_op_{uid}", f"op_val_{uid}"
    min_slider_id, min_val_id = f"sl_min_{uid}", f"min_val_{uid}"
    max_slider_id, max_val_id = f"sl_max_{uid}", f"max_val_{uid}"

    shape_js = json.dumps(shape)
    bg_filename = img_path.name

    return HTML(f"""
    <style>
        .nv-container {{
            width: {width + 200}px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f1115;
            color: #e1e4e8;
            border-radius: 10px;
            overflow: hidden;
            box-sizing: border-box;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
        }}
        .nv-toolbar {{
            background: rgba(22, 25, 32, 0.95);
            display: flex;
            align-items: center;
            padding: 0 10px;
            height: 38px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            gap: 8px;
        }}
        .nv-btn {{
            background: transparent;
            color: #c9d1d9;
            border: 1px solid transparent;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 500;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .nv-btn:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
        }}
        .nv-dropdown {{
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            background: #161920;
            min-width: 120px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6);
            z-index: 1000;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 4px;
            margin-top: 4px;
        }}
        .nv-menu-item {{
            padding: 6px 10px;
            cursor: pointer;
            font-size: 12px;
            border-radius: 4px;
            color: #8b949e;
            transition: background 0.12s ease;
        }}
        .nv-menu-item:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
        }}
        .nv-menu-item.active {{
            color: #58a6ff;
            font-weight: 600;
        }}
        .nv-main-layout {{
            display: flex;
            width: 100%;
        }}
        .nv-canvas-wrap {{
            width: {width}px;
            height: {height}px;
            background: #000000;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        .nv-sidebar {{
            width: 200px;
            background: #161920;
            border-left: 1px solid rgba(255, 255, 255, 0.06);
            padding: 12px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 14px;
            font-size: 11px;
            color: #8b949e;
        }}
        .nv-sidebar-section {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .nv-section-title {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #484f58;
            font-weight: 700;
            margin-bottom: 2px;
        }}
        .nv-slider-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
        }}
        .nv-slider {{
            -webkit-appearance: none;
            width: 80px;
            height: 4px;
            border-radius: 2px;
            background: #30363d;
            outline: none;
        }}
        .nv-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #58a6ff;
            cursor: pointer;
        }}
        .nv-val-highlight {{
            color: #58a6ff;
            font-weight: 600;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
        }}
        .nv-file-info {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 10px;
            color: #c9d1d9;
            word-break: break-all;
            background: rgba(0, 0, 0, 0.2);
            padding: 4px 6px;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }}
    </style>

    <div class="nv-container">
        <div class="nv-toolbar">
            <div style="position:relative;">
                <button type="button" class="nv-btn" id="{btn_view_id}">View ▾</button>
                <div class="nv-dropdown" id="{dropdown_view_id}">
                    <div class="nv-menu-item active" data-mode="coronal">Coronal</div>
                    <div class="nv-menu-item" data-mode="axial">Axial</div>
                    <div class="nv-menu-item" data-mode="sagittal">Sagittal</div>
                </div>
            </div>

            <div style="position:relative; {'display:none;' if not has_overlay else ''}">
                <button type="button" class="nv-btn" id="{btn_color_id}">Color ▾</button>
                <div class="nv-dropdown" id="{dropdown_color_id}">
                    <div class="nv-menu-item active" data-cmap="red">Red</div>
                    <div class="nv-menu-item" data-cmap="green">Green</div>
                    <div class="nv-menu-item" data-cmap="blue">Blue</div>
                    <div class="nv-menu-item" data-cmap="hot">Hot</div>
                    <div class="nv-menu-item" data-cmap="cool">Cool</div>
                    <div class="nv-menu-item" data-cmap="rainbow">Rainbow</div>
                </div>
            </div>
        </div>

        <div class="nv-main-layout">
            <div class="nv-canvas-wrap">
                <canvas id="{canvas_id}" width="{width}" height="{height}" style="cursor:crosshair;"></canvas>
            </div>

            <div class="nv-sidebar">
                <div class="nv-sidebar-section" style="{'display:none;' if not has_overlay else ''}">
                    <div class="nv-section-title">Overlay Controls</div>
                    <div class="nv-slider-row">
                        <span>Opacity</span>
                        <input type="range" class="nv-slider" id="{op_slider_id}" min="0" max="1" step="0.05" value="0.5">
                        <span id="{op_val_id}" class="nv-val-highlight" style="width:28px;">0.50</span>
                    </div>
                    <div class="nv-slider-row">
                        <span>Min</span>
                        <input type="range" class="nv-slider" id="{min_slider_id}" min="{ov_min}" max="{ov_max}" step="{(ov_max - ov_min) / 100 or 0.01}" value="{p5}">
                        <span id="{min_val_id}" class="nv-val-highlight" style="width:28px;">{p5:.1f}</span>
                    </div>
                    <div class="nv-slider-row">
                        <span>Max</span>
                        <input type="range" class="nv-slider" id="{max_slider_id}" min="{ov_min}" max="{ov_max}" step="{(ov_max - ov_min) / 100 or 0.01}" value="{p95}">
                        <span id="{max_val_id}" class="nv-val-highlight" style="width:28px;">{p95:.1f}</span>
                    </div>
                </div>

                <div class="nv-sidebar-section">
                    <div class="nv-section-title">Cursor Info</div>
                    <div id="{info_id}" style="display:flex; flex-direction:column; gap:4px;">
                        <div>Voxel: <span class="nv-val-highlight">[-, -, -]</span></div>
                        <div>Intensity: <span class="nv-val-highlight">-</span></div>
                    </div>
                </div>

                <div class="nv-sidebar-section" style="margin-top:auto;">
                    <div class="nv-section-title">Loaded Files</div>
                    <div style="display:flex; flex-direction:column; gap:4px;">
                        <div>BG: <div class="nv-file-info">{bg_filename}</div></div>
                        <div style="{'display:none;' if not has_overlay else ''}">OV: <div class="nv-file-info">{overlay_filename}</div></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
    (async function() {{
        const shape = {shape_js};
        const nx = shape[0], ny = shape[1], nz = shape[2];
        const baseMin = {d_min}, baseMax = {d_max};
        const ovAbsMin = {ov_min}, ovAbsMax = {ov_max};

        async function decompressZlib(b64) {{
            const bin = atob(b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

            const cs = new DecompressionStream("deflate");
            const writer = cs.writable.getWriter();
            writer.write(bytes);
            writer.close();

            const buffer = await new Response(cs.readable).arrayBuffer();
            return new Uint8Array(buffer);
        }}

        const baseVol = await decompressZlib("{base_b64}");
        const hasOverlay = {str(has_overlay).lower()};
        const overlayVol = hasOverlay ? await decompressZlib("{overlay_b64}") : null;

        let viewMode = "coronal";
        let activeCmap = "{colormap.lower()}";
        let overlayOpacity = 0.5;
        let ovThreshMin = {p5};
        let ovThreshMax = {p95};
        let currentVox = [Math.floor(nx / 2), Math.floor(ny / 2), Math.floor(nz / 2)];

        const canvas = document.getElementById("{canvas_id}");
        const ctx = canvas.getContext("2d");
        const infoEl = document.getElementById("{info_id}");

        function getVoxelIdx(i, j, k) {{
            return (i * ny * nz) + (j * nz) + k;
        }}

        function getColormapRGB(val, cmap) {{
            const v = Math.min(1.0, Math.max(0.0, val));
            const c8 = Math.floor(v * 255);
            if (cmap === "green") return [0, c8, 0];
            if (cmap === "blue") return [0, 0, c8];
            if (cmap === "hot") {{
                return [
                    Math.min(255, Math.floor(v * 3 * 255)),
                    Math.min(255, Math.max(0, Math.floor((v - 0.33) * 3 * 255))),
                    Math.min(255, Math.max(0, Math.floor((v - 0.66) * 3 * 255)))
                ];
            }}
            if (cmap === "cool") return [Math.floor((1 - v) * 255), Math.floor(v * 255), 255];
            if (cmap === "rainbow") {{
                const h = (1 - v) * 240;
                const f = (h / 60) % 1;
                const q = Math.floor(255 * (1 - f)), t = Math.floor(255 * f);
                if (h < 60) return [255, t, 0];
                if (h < 120) return [q, 255, 0];
                if (h < 180) return [0, 255, t];
                return [0, q, 255];
            }}
            return [c8, 0, 0];
        }}

        function drawSlice() {{
            const cw = canvas.width;
            const ch = canvas.height;
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, cw, ch);

            let sw, sh;
            if (viewMode === "axial") {{ sw = nx; sh = ny; }}
            else if (viewMode === "coronal") {{ sw = nx; sh = nz; }}
            else {{ sw = ny; sh = nz; }}

            const imgData = ctx.createImageData(sw, sh);
            const data = imgData.data;

            let pxIdx = 0;
            for (let row = sh - 1; row >= 0; row--) {{
                for (let col = 0; col < sw; col++) {{
                    let i, j, k;
                    if (viewMode === "axial") {{ i = col; j = row; k = currentVox[2]; }}
                    else if (viewMode === "coronal") {{ i = col; j = currentVox[1]; k = row; }}
                    else {{ i = currentVox[0]; j = col; k = row; }}

                    const vIdx = getVoxelIdx(i, j, k);
                    const baseVal = baseVol[vIdx];

                    let r = baseVal, g = baseVal, b = baseVal;

                    if (hasOverlay && overlayVol) {{
                        const ovNormByte = overlayVol[vIdx];
                        const ovFloatVal = ovAbsMin + (ovNormByte / 255.0) * (ovAbsMax - ovAbsMin);

                        if (ovFloatVal >= ovThreshMin && ovThreshMax > ovThreshMin) {{
                            const normOv = (ovFloatVal - ovThreshMin) / (ovThreshMax - ovThreshMin);
                            const [or, og, ob] = getColormapRGB(normOv, activeCmap);
                            const a = overlayOpacity;
                            r = Math.floor(r * (1 - a) + or * a);
                            g = Math.floor(g * (1 - a) + og * a);
                            b = Math.floor(b * (1 - a) + ob * a);
                        }}
                    }}

                    data[pxIdx] = r;
                    data[pxIdx + 1] = g;
                    data[pxIdx + 2] = b;
                    data[pxIdx + 3] = 255;

                    pxIdx += 4;
                }}
            }}

            const offscreen = document.createElement("canvas");
            offscreen.width = sw;
            offscreen.height = sh;
            offscreen.getContext("2d").putImageData(imgData, 0, 0);

            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(offscreen, 0, 0, cw, ch);

            ctx.strokeStyle = "rgba(88, 166, 255, 0.85)";
            ctx.lineWidth = 1;

            let cx, cy;
            if (viewMode === "axial") {{
                cx = (currentVox[0] / sw) * cw;
                cy = ((sh - 1 - currentVox[1]) / sh) * ch;
            }} else if (viewMode === "coronal") {{
                cx = (currentVox[0] / sw) * cw;
                cy = ((sh - 1 - currentVox[2]) / sh) * ch;
            }} else {{
                cx = (currentVox[1] / sw) * cw;
                cy = ((sh - 1 - currentVox[2]) / sh) * ch;
            }}

            ctx.beginPath();
            ctx.moveTo(cx, 0); ctx.lineTo(cx, ch);
            ctx.moveTo(0, cy); ctx.lineTo(cw, cy);
            ctx.stroke();

            updateInfo();
        }}

        function updateInfo() {{
            const [i, j, k] = currentVox;
            const idx = getVoxelIdx(i, j, k);
            const val = baseMin + (baseVol[idx] / 255.0) * (baseMax - baseMin);
            infoEl.innerHTML = `
                <div>Voxel: <span class="nv-val-highlight">[${{i}}, ${{j}}, ${{k}}]</span></div>
                <div>Intensity: <span class="nv-val-highlight">${{val.toFixed(4)}}</span></div>
            `;
        }}

        canvas.addEventListener("mousedown", (e) => {{
            const rect = canvas.getBoundingClientRect();
            const clickX = (e.clientX - rect.left) / canvas.width;
            const clickY = (e.clientY - rect.top) / canvas.height;

            if (viewMode === "axial") {{
                currentVox[0] = Math.min(nx - 1, Math.max(0, Math.floor(clickX * nx)));
                currentVox[1] = Math.min(ny - 1, Math.max(0, Math.floor((1 - clickY) * ny)));
            }} else if (viewMode === "coronal") {{
                currentVox[0] = Math.min(nx - 1, Math.max(0, Math.floor(clickX * nx)));
                currentVox[2] = Math.min(nz - 1, Math.max(0, Math.floor((1 - clickY) * nz)));
            }} else {{
                currentVox[1] = Math.min(ny - 1, Math.max(0, Math.floor(clickX * ny)));
                currentVox[2] = Math.min(nz - 1, Math.max(0, Math.floor((1 - clickY) * nz)));
            }}
            drawSlice();
        }});

        canvas.addEventListener("wheel", (e) => {{
            e.preventDefault();
            const delta = e.deltaY > 0 ? -1 : 1;

            if (viewMode === "axial") {{
                currentVox[2] = Math.min(nz - 1, Math.max(0, currentVox[2] + delta));
            }} else if (viewMode === "coronal") {{
                currentVox[1] = Math.min(ny - 1, Math.max(0, currentVox[1] + delta));
            }} else {{
                currentVox[0] = Math.min(nx - 1, Math.max(0, currentVox[0] + delta));
            }}
            drawSlice();
        }}, {{ passive: false }});

        const btnView = document.getElementById("{btn_view_id}");
        const ddView = document.getElementById("{dropdown_view_id}");
        
        btnView.addEventListener("click", (e) => {{
            e.stopPropagation();
            ddView.style.display = ddView.style.display === "block" ? "none" : "block";
            if (hasOverlay) document.getElementById("{dropdown_color_id}").style.display = "none";
        }});

        ddView.querySelectorAll(".nv-menu-item").forEach(item => {{
            item.addEventListener("click", () => {{
                viewMode = item.getAttribute("data-mode");
                ddView.querySelectorAll(".nv-menu-item").forEach(i => i.classList.remove("active"));
                item.classList.add("active");
                ddView.style.display = "none";
                drawSlice();
            }});
        }});

        if (hasOverlay) {{
            const btnColor = document.getElementById("{btn_color_id}");
            const ddColor = document.getElementById("{dropdown_color_id}");

            btnColor.addEventListener("click", (e) => {{
                e.stopPropagation();
                ddColor.style.display = ddColor.style.display === "block" ? "none" : "block";
                ddView.style.display = "none";
            }});

            ddColor.querySelectorAll(".nv-menu-item").forEach(item => {{
                item.addEventListener("click", () => {{
                    activeCmap = item.getAttribute("data-cmap");
                    ddColor.querySelectorAll(".nv-menu-item").forEach(i => i.classList.remove("active"));
                    item.classList.add("active");
                    ddColor.style.display = "none";
                    drawSlice();
                }});
            }});

            document.getElementById("{op_slider_id}").addEventListener("input", (e) => {{
                overlayOpacity = parseFloat(e.target.value);
                document.getElementById("{op_val_id}").innerText = overlayOpacity.toFixed(2);
                drawSlice();
            }});

            document.getElementById("{min_slider_id}").addEventListener("input", (e) => {{
                ovThreshMin = parseFloat(e.target.value);
                document.getElementById("{min_val_id}").innerText = ovThreshMin.toFixed(1);
                drawSlice();
            }});

            document.getElementById("{max_slider_id}").addEventListener("input", (e) => {{
                ovThreshMax = parseFloat(e.target.value);
                document.getElementById("{max_val_id}").innerText = ovThreshMax.toFixed(1);
                drawSlice();
            }});
        }}

        document.addEventListener("click", () => {{
            ddView.style.display = "none";
            if (hasOverlay) document.getElementById("{dropdown_color_id}").style.display = "none";
        }});

        drawSlice();
    }})();
    </script>
    """)