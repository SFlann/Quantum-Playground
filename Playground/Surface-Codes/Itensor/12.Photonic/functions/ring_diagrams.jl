# ============================================================================
#  ring_diagrams.jl — SVG diagrams of the ring fusion network, generated from
#  the SAME pairing tables the machine runs on (passed in as arguments, so the
#  pictures can never drift from the code). Pure drawing helpers — no physics.
#
#  Conventions: TIME FLOWS UP.  z− (n1) at the bottom of each ring (the world-
#  line arrives there), z+ (n4) at the top (it leaves there); arm 1 (n2, n3) on
#  the right, arm 2 (n6, n5) on the left; `mirror=true` flips a ring left-right
#  (waveguide layout is arbitrary — only the fusion GRAPH matters).
#  Display in the notebook with  show_svg(...)  (image/svg+xml MIME).
# ============================================================================

show_svg(s::String) = display(MIME("image/svg+xml"), s)

svg_open(w, h) = "<svg xmlns='http://www.w3.org/2000/svg' width='$w' height='$h' viewBox='0 0 $w $h'>" *
                 "<style>text{font-family:Helvetica,Arial,sans-serif}</style>" *
                 "<rect width='$w' height='$h' fill='white'/>"
circ(x, y, r, fill; stroke="#333", sw=1.3) =
    "<circle cx='$(round(x,digits=1))' cy='$(round(y,digits=1))' r='$r' fill='$fill' stroke='$stroke' stroke-width='$sw'/>"
lin(x1, y1, x2, y2; stroke="#444", sw=1.4, dash="") =
    "<line x1='$(round(x1,digits=1))' y1='$(round(y1,digits=1))' x2='$(round(x2,digits=1))' y2='$(round(y2,digits=1))' " *
    "stroke='$stroke' stroke-width='$sw'$(dash == "" ? "" : " stroke-dasharray='$dash'")/>"
txt(x, y, s; size=12, fill="#222", anchor="middle", weight="normal") =
    "<text x='$(round(x,digits=1))' y='$(round(y,digits=1))' font-size='$size' fill='$fill' text-anchor='$anchor' font-weight='$weight'>$s</text>"
poly(pts, fill; opacity=0.3) =
    "<polygon points='" * join(["$(round(x,digits=1)),$(round(y,digits=1))" for (x,y) in pts], " ") *
    "' fill='$fill' opacity='$opacity' stroke='none'/>"

function fusionlink(x1, y1, x2, y2; color="#2166ac", label="", lx=0, ly=-9, sw=2.2)
    mx, my = (x1+x2)/2, (y1+y2)/2
    dx, dy = x2-x1, y2-y1; L = max(sqrt(dx^2+dy^2), 1e-9); ux, uy = dx/L, dy/L
    px, py = -uy, ux; b = 6.5
    s = lin(x1, y1, x2, y2; stroke=color, sw=sw)
    s *= "<polygon points='$(round(mx-b*ux+b*px,digits=1)),$(round(my-b*uy+b*py,digits=1)) " *
         "$(round(mx-b*ux-b*px,digits=1)),$(round(my-b*uy-b*py,digits=1)) " *
         "$(round(mx+b*ux+b*px,digits=1)),$(round(my+b*uy+b*py,digits=1)) " *
         "$(round(mx+b*ux-b*px,digits=1)),$(round(my+b*uy-b*py,digits=1))' fill='$color'/>"
    label != "" && (s *= txt(mx+lx, my+ly, label; size=11, fill=color, weight="bold"))
    return s
end

const RING_ANGLE = Dict(1 => 90, 2 => 30, 3 => -30, 4 => -90, 5 => -150, 6 => 150)
const ROLECOL = Dict(:zin => "#1a9850", :zout => "#e08214", :fuseXX => "#2166ac",
                     :fuseZZ => "#7b3294", :pad => "#a6a6a6", :prune => "#dddddd")

slotxy(cx, cy, R, k; mirror=false) = begin
    a = RING_ANGLE[k] * pi / 180
    (cx + R * cos(a) * (mirror ? -1 : 1), cy + R * sin(a))
end
leafxy(cx, cy, R, k; d=22, mirror=false) = begin
    a = RING_ANGLE[k] * pi / 180
    (cx + (R + d) * cos(a) * (mirror ? -1 : 1), cy + (R + d) * sin(a))
end

function draw_ring(cx, cy; R=34, roles=Dict{Int,Symbol}(), labels=true, noder=8,
                   mirror=false, dead=Int[], name="")
    s = ""
    pts = [slotxy(cx, cy, R, k; mirror=mirror) for k in 1:6]
    for k in 1:6
        j = k == 6 ? 1 : k + 1
        s *= lin(pts[k]..., pts[j]...; stroke="#777", sw=1.6)
    end
    for k in 1:6
        role = get(roles, k, :pad)
        col = get(ROLECOL, role, "#a6a6a6")
        (x, y) = pts[k]
        if role in (:zin, :zout, :fuseXX, :fuseZZ) && !(k in dead)
            (lx, ly) = leafxy(cx, cy, R, k; mirror=mirror)
            s *= lin(x, y, lx, ly; stroke=col, sw=1.4)
            s *= circ(lx, ly, 5.2, "white"; stroke=col, sw=1.8)
        end
        s *= circ(x, y, noder, k in dead ? "#eeeeee" : col; stroke=k in dead ? "#bbb" : "#333")
        if k in dead
            s *= txt(x, y + 3.5, "✕"; size=10, fill="#999")
        elseif labels
            s *= txt(x, y + 3.5, "n$k"; size=9, fill=role == :prune ? "#888" : "white", weight="bold")
        end
    end
    name != "" && (s *= txt(cx, cy + 4, name; size=13, weight="bold", fill="#333"))
    return s
end

# ======================================================================
# A — one ring, photon by photon (d5 on an odd slab)
# ======================================================================
function diagramA()
    s = svg_open(820, 480)
    cx, cy, R = 285, 250, 80
    roles = Dict(1 => :zin, 4 => :zout, 3 => :fuseXX, 6 => :fuseZZ, 2 => :pad, 5 => :prune)
    s *= draw_ring(cx, cy; R=R, roles=roles, noder=13)
    s *= txt(cx, cy - R - 52, "z⁺ (n4): the worldline LEAVES here —"; size=12, fill=ROLECOL[:zout], weight="bold")
    s *= txt(cx, cy - R - 38, "its leaf waits for the NEXT slab's fusion"; size=12, fill=ROLECOL[:zout])
    s *= txt(cx, cy + R + 46, "z⁻ (n1): the worldline ARRIVES —"; size=12, fill=ROLECOL[:zin], weight="bold")
    s *= txt(cx, cy + R + 60, "one fusion with last slab's flying leaf"; size=12, fill=ROLECOL[:zin])
    s *= txt(cx + R + 44, cy - 56, "n3: cell fusion,"; size=12, fill=ROLECOL[:fuseXX], weight="bold", anchor="start")
    s *= txt(cx + R + 44, cy - 42, "read on the XX leg"; size=12, fill=ROLECOL[:fuseXX], anchor="start")
    s *= txt(cx + R + 44, cy + 44, "n2: PAD — no fusion;"; size=12, fill="#777", anchor="start")
    s *= txt(cx + R + 44, cy + 58, "X-measured = a wire hop"; size=12, fill="#777", anchor="start")
    s *= txt(cx - R - 44, cy - 56, "n5: PRUNE — deleted by"; size=12, fill="#999", anchor="end")
    s *= txt(cx - R - 44, cy - 42, "a single-photon Z"; size=12, fill="#999", anchor="end")
    s *= txt(cx - R - 44, cy + 44, "n6: cell fusion,"; size=12, fill=ROLECOL[:fuseZZ], weight="bold", anchor="end")
    s *= txt(cx - R - 44, cy + 58, "read on the ZZ leg"; size=12, fill=ROLECOL[:fuseZZ], anchor="end")
    lx, ly = 590, 160
    s *= txt(lx + 95, ly - 28, "the ring of d5, odd slab"; size=14, weight="bold")
    items = [(ROLECOL[:zin], "z⁻ — temporal fusion in"),
             (ROLECOL[:zout], "z⁺ — temporal fusion out"),
             (ROLECOL[:fuseXX], "cell fusion — XX leg = check edge"),
             (ROLECOL[:fuseZZ], "cell fusion — ZZ leg = check edge"),
             ("#a6a6a6", "pad — pattern X (wire)"),
             ("#dddddd", "prune — single-photon Z (delete)")]
    for (i, (col, lab)) in enumerate(items)
        y = ly + 26i
        s *= circ(lx, y, 7, col; stroke="#333")
        s *= txt(lx + 15, y + 4, lab; size=11, anchor="start")
    end
    s *= txt(lx + 95, ly + 26*7 + 4, "small white circles = the LEAVES,"; size=11, fill="#555")
    s *= txt(lx + 95, ly + 26*7 + 18, "the photons consumed in fusions"; size=11, fill="#555")
    s * "</svg>"
end

# ======================================================================
# B — the 3×3 patch: rings + fusion links (generated from the real tables)
# ======================================================================
slotnum(s) = parse(Int, string(s)[2:2])

function diagramB(tab, title, sub, cellcol, shade)
    W, H = 700, 640
    s = svg_open(W, H)
    ringat(i) = (140 + 190 * ((i-1) % 3), 140 + 190 * ((i-1) ÷ 3))
    col(i) = (i-1) % 3
    # auto-mirror: a ring is flipped when a fused slot faces away from its partner
    mirror = Dict(i => false for i in 1:9)
    for (cell, a, sa, b, sb) in tab
        for (q, sq, p) in ((a, sa, b), (b, sb, a))
            dx = col(p) - col(q)
            dx == 0 && continue
            k = slotnum(sq)
            ((k in (2,3) && dx < 0) || (k in (5,6) && dx > 0)) && (mirror[q] = true)
        end
    end
    for (cell, sup) in shade                      # the cells this slab READS, shaded
        if length(sup) == 4
            s *= poly([ringat(sup[1]), ringat(sup[2]), ringat(sup[4]), ringat(sup[3])], cellcol; opacity=0.13)
        else
            (x1,y1) = ringat(sup[1]); (x2,y2) = ringat(sup[2])
            px, py = -(y2-y1), (x2-x1); n = sqrt(px^2+py^2); px, py = 28px/n, 28py/n
            s *= poly([(x1+px,y1+py),(x2+px,y2+py),(x2-px,y2-py),(x1-px,y1-py)], cellcol; opacity=0.13)
        end
    end
    fused = Dict{Int,Vector{Int}}()
    for (cell, a, sa, b, sb) in tab               # links from the ACTUAL pairing table
        push!(get!(fused, a, Int[]), slotnum(sa)); push!(get!(fused, b, Int[]), slotnum(sb))
        (ax, ay) = ringat(a); (bx, by) = ringat(b)
        (lax, lay) = leafxy(ax, ay, 34, slotnum(sa); d=13, mirror=mirror[a])
        (lbx, lby) = leafxy(bx, by, 34, slotnum(sb); d=13, mirror=mirror[b])
        vert = abs(lax - lbx) < abs(lay - lby)
        s *= fusionlink(lax, lay, lbx, lby; color=cellcol, label=cell, lx=(vert ? 30 : 0), ly=(vert ? 4 : -10))
    end
    for i in 1:9
        (x, y) = ringat(i)
        roles = Dict{Int,Symbol}(1 => :zin, 4 => :zout, 2 => :pad, 5 => :prune)
        for k in get(fused, i, Int[])
            roles[k] = k in (2, 3) ? :fuseXX : :fuseZZ
        end
        haskey(roles, 3) || (roles[3] = :pad)     # unfused arm-1 slot → pad
        haskey(roles, 6) || (roles[6] = :prune)   # unfused arm-2 slot → prune
        s *= draw_ring(x, y; R=34, roles=roles, noder=7.5, labels=false, mirror=mirror[i], name="d$i")
    end
    s *= txt(W/2, 34, title; size=15, weight="bold")
    s *= txt(W/2, 54, sub; size=11.5, fill="#555")
    s * "</svg>"
end

# ======================================================================
# C — space AND time: the (d4,d5) pair over three slabs
# ======================================================================
function diagramC()
    W, H = 760, 880
    s = svg_open(W, H)
    xs = (225, 520)
    ys = (680, 450, 220)                # slabs 1,2,3 — time flows up
    R = 42
    cellname = ("Z1245", "X4578", "Z1245")
    reads    = ("reads  Z₄Z₅  — a primal (Z-cell) edge", "reads  X₄X₅  — a dual (X-cell) edge", "reads  Z₄Z₅  — a primal (Z-cell) edge")
    cellcol  = ("#2166ac", "#c51b7d", "#2166ac")
    for (ti, y) in enumerate(ys)
        # d4 default (its n3 faces right, towards d5); d5 mirrored so its n3 faces LEFT
        roles4 = Dict(1 => :zin, 4 => :zout, 3 => :fuseXX, 6 => (isodd(ti) ? :fuseZZ : :prune), 2 => :pad, 5 => :prune)
        roles5 = Dict(1 => :zin, 4 => :zout, 3 => :fuseXX, 6 => :fuseZZ, 2 => :pad, 5 => :prune)
        s *= draw_ring(xs[1], y; R=R, roles=roles4, noder=9)
        s *= draw_ring(xs[2], y; R=R, roles=roles5, noder=9, mirror=true)
        s *= txt(xs[1] - R - 24, y - R - 14, "d4, slab $ti"; size=11.5, fill="#333", anchor="end")
        s *= txt(xs[2] + R + 24, y - R - 14, "d5, slab $ti"; size=11.5, fill="#333", anchor="start")
        # the (4,5) pairing this slab: n3 ⋈ n3, facing each other
        (l4x, l4y) = leafxy(xs[1], y, R, 3)
        (l5x, l5y) = leafxy(xs[2], y, R, 3; mirror=true)
        s *= fusionlink(l4x, l4y, l5x, l5y; color=cellcol[ti], label=cellname[ti], ly=-12)
        s *= txt((xs[1]+xs[2])/2, y + 34, reads[ti]; size=11.5, fill=cellcol[ti])
        # stubs for the fusions leaving the picture (to d7/d6's rings)
        if isodd(ti)
            (sx, sy) = leafxy(xs[1], y, R, 6)
            s *= lin(sx, sy, sx - 40, sy; stroke="#7b3294", sw=1.6, dash="4 3")
            s *= txt(sx - 46, sy + 4, "to d7"; size=10, fill="#7b3294", anchor="end")
        end
        (sx, sy) = leafxy(xs[2], y, R, 6; mirror=true)
        s *= lin(sx, sy, sx + 40, sy; stroke="#7b3294", sw=1.6, dash="4 3")
        s *= txt(sx + 46, sy + 4, "to d6"; size=10, fill="#7b3294", anchor="start")
    end
    for x in xs, ti in 1:2                       # temporal fusions between slabs
        m = (x == xs[2])
        (ax, ay) = leafxy(x, ys[ti],   R, 4; mirror=m)
        (bx, by) = leafxy(x, ys[ti+1], R, 1; mirror=m)
        s *= fusionlink(ax, ay, bx, by; color="#1a9850",
                        label=(x == xs[2] && ti == 1 ? "temporal fusion" : ""), lx=92, ly=4)
    end
    for x in xs                                  # the input photons entering slab 1
        m = (x == xs[2])
        (bx, by) = leafxy(x, ys[1], R, 1; mirror=m)
        s *= fusionlink(x, ys[1] + R + 88, bx, by; color="#1a9850")
        s *= circ(x, ys[1] + R + 96, 7, "#444")
        s *= txt(x, ys[1] + R + 118, "input photon (|0_L⟩ layer)"; size=10.5, fill="#444")
    end
    for ti in 1:2
        ymid = (ys[ti] + ys[ti+1]) / 2
        s *= txt(64, ymid - 2, "1 H per"; size=11, fill="#888")
        s *= txt(64, ymid + 11, "worldline"; size=11, fill="#888")
    end
    s *= txt(W/2, 36, "the SAME hardware, slab after slab —"; size=14, weight="bold")
    s *= txt(W/2, 54, "the per-slab Hadamard frame alternates what the fusions READ"; size=12.5, fill="#333")
    s *= txt(W/2, H - 16, "time flows upward · green ⋈ = temporal fusions (the advance) · blue/pink ⋈ = check edges"; size=11, fill="#555")
    s * "</svg>"
end

# ======================================================================
# C2 — 3D: the whole patch, three slabs stacked in time (isometric)
# ======================================================================
function diagramC2(tab_odd, tab_even)
    W, H = 780, 880
    s = svg_open(W, H)
    R = 17
    pos(i, layer) = begin
        c = (i-1) % 3; r = (i-1) ÷ 3
        (120 + 165c + 62r, 760 - 250*(layer-1) - 46r)
    end
    for layer in 1:3                             # bottom (earliest) first
        tab = isodd(layer) ? tab_odd : tab_even
        colr = isodd(layer) ? "#2166ac" : "#c51b7d"
        if layer > 1                             # temporal links to the layer below
            for i in 1:9
                (x1, y1) = pos(i, layer - 1); (x2, y2) = pos(i, layer)
                s *= fusionlink(x1, y1 - R - 2, x2, y2 + R + 2; color="#1a9850", sw=1.5)
            end
        end
        for (cell, a, sa, b, sb) in tab          # the slab's spatial fusions
            (ax, ay) = pos(a, layer); (bx, by) = pos(b, layer)
            dx, dy = bx-ax, by-ay; L = sqrt(dx^2+dy^2); ux, uy = dx/L, dy/L
            s *= fusionlink(ax + (R+4)*ux, ay + (R+4)*uy, bx - (R+4)*ux, by - (R+4)*uy; color=colr, sw=1.8)
        end
        for i in 1:9                             # rings as small hexagons
            (x, y) = pos(i, layer)
            pts = [slotxy(x, y, R, k) for k in 1:6]
            for k in 1:6
                s *= lin(pts[k]..., pts[k == 6 ? 1 : k+1]...; stroke="#888", sw=1.2)
            end
            for k in 1:6
                s *= circ(pts[k][1], pts[k][2], 3.1, k == 1 ? ROLECOL[:zin] : k == 4 ? ROLECOL[:zout] : "#bbb"; stroke="#555", sw=0.8)
            end
        end
        (lx, ly) = pos(3, layer)
        s *= txt(lx + 78, ly + 8, "slab $layer"; size=13, weight="bold", fill=colr, anchor="start")
        s *= txt(lx + 78, ly + 24, isodd(layer) ? "reads the Z-cells" : "reads the X-cells"; size=11, fill=colr, anchor="start")
    end
    s *= txt(W/2, 36, "the fusion network IS spacetime"; size=15, weight="bold")
    s *= txt(W/2, 56, "9 rings per slab · green ⋈ = temporal fusions carry the worldlines up · every fusion is local"; size=11.5, fill="#555")
    s * "</svg>"
end

# ======================================================================
# D — the life of one ring (vertical strip): delivered → retired → replaced
# ======================================================================
function diagramD()
    W, H = 640, 1150
    s = svg_open(W, H)
    R = 46; cx = 330
    roles = Dict(1 => :zin, 4 => :zout, 3 => :fuseXX, 6 => :fuseZZ, 2 => :pad, 5 => :prune)
    # -- panel 1
    cy = 215
    s *= txt(cx, 58, "1 · delivered and fused"; size=14, weight="bold")
    s *= txt(cx, 76, "the two beam-splitter types consume the LEAVES: one temporal ⋈ (arrive),"; size=11, fill="#555")
    s *= txt(cx, 90, "one ⋈ per cell the qubit belongs to; pads X-measured, prunes Z-deleted"; size=11, fill="#555")
    s *= draw_ring(cx, cy; R=R, roles=roles, noder=9)
    (l1x, l1y) = leafxy(cx, cy, R, 1)
    s *= fusionlink(l1x, l1y, cx, cy + R + 76; color="#1a9850")
    s *= circ(cx, cy + R + 84, 6.5, ROLECOL[:zout])
    s *= txt(cx + 14, cy + R + 88, "flying photon from slab t−1"; size=10.5, fill="#444", anchor="start")
    for (k, colr, lab) in ((3, "#2166ac", "to d4's ring"), (6, "#7b3294", "to d6's ring"))
        (lx, ly) = leafxy(cx, cy, R, k)
        ex = k == 3 ? lx + 70 : lx - 70
        s *= fusionlink(lx, ly, ex, ly; color=colr)
        s *= txt(ex + (k == 3 ? 6 : -6), ly + 4, lab; size=10.5, fill=colr, anchor=k == 3 ? "start" : "end")
    end
    # -- panel 2
    cy = 600
    s *= txt(cx, 445, "2 · retired — all its fusions are done"; size=14, weight="bold")
    s *= txt(cx, 463, "the fused-slot NODES are pattern-measured (X) and dropped from the register;"; size=11, fill="#555")
    s *= txt(cx, 477, "greyed ✕ = measured and gone. ALL that survives of qubit i:"; size=11, fill="#555")
    s *= txt(cx, 495, "the z⁺ photon + its leaf  =  the new flying photon (the worldline)"; size=11.5, fill=ROLECOL[:zout], weight="bold")
    s *= draw_ring(cx, cy; R=R, roles=roles, noder=9, dead=[1, 2, 3, 5, 6])
    # -- panel 3
    cy = 990
    s *= txt(cx, 810, "3 · slab t+1 arrives — repeat"; size=14, weight="bold")
    s *= txt(cx, 828, "a fresh ring's z⁻ leaf fuses with the flying leaf: the worldline"; size=11, fill="#555")
    s *= txt(cx, 842, "enters the next ring; the old z⁺ node is X-measured and dropped"; size=11, fill="#555")
    s *= draw_ring(cx, cy - 45; R=R, roles=roles, noder=9)
    (zx, zy) = leafxy(cx, cy - 45, R, 1)
    s *= fusionlink(zx, zy, cx, cy + R + 40; color="#1a9850")
    s *= circ(cx, cy + R + 48, 6.5, ROLECOL[:zout])
    s *= txt(cx + 14, cy + R + 52, "slab t's flying photon"; size=10.5, fill="#444", anchor="start")
    s * "</svg>"
end

nothing
