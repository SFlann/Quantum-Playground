# ============================================================================
# canonical_ring_diagrams.jl
#
# SVG teaching diagrams for the canonical six-photon 6-ring fusion network.
# Reuses the drawing primitives from ring_diagrams.jl, but deliberately does
# NOT draw an extra leaf qubit. Each colored disc is one of the six photons.
# A line leaving a disc is an optical route to a destructive fusion.
# ============================================================================

const CANON_PORTCOL = Dict(
    1 => "#1a9850",  # t-
    2 => "#7b3294",  # y+
    3 => "#2166ac",  # x+
    4 => "#e08214",  # t+
    5 => "#7b3294",  # y-
    6 => "#2166ac",  # x-
)
const CANON_PORTLAB = Dict(1=>"t⁻", 2=>"y⁺", 3=>"x⁺", 4=>"t⁺", 5=>"y⁻", 6=>"x⁻")

function canon_draw_ring(cx, cy; R=34, labels=true, name="", noder=8,
                         faded=Int[], route=Int[], boundary=Int[])
    s = ""
    pts = [slotxy(cx, cy, R, k) for k in 1:6]
    for k in 1:6
        j = k == 6 ? 1 : k + 1
        s *= lin(pts[k]..., pts[j]...; stroke="#666", sw=1.7)
    end
    for k in 1:6
        x, y = pts[k]
        col = k in faded ? "#dddddd" : CANON_PORTCOL[k]
        if k in route || k in boundary
            ox, oy = leafxy(cx, cy, R, k; d=17)
            s *= lin(x, y, ox, oy; stroke=col, sw=1.5, dash=(k in boundary ? "3 2" : ""))
            if k in boundary
                s *= txt(ox, oy+4, "Z"; size=10, fill="#777", weight="bold")
            end
        end
        s *= circ(x, y, noder, col; stroke="#333", sw=1.3)
        labels && (s *= txt(x, y+3.3, CANON_PORTLAB[k]; size=8.5, fill="white", weight="bold"))
    end
    name != "" && (s *= txt(cx, cy+4, name; size=12, fill="#333", weight="bold"))
    return s
end

function canon_diagramA()
    W,H = 900,500
    s = svg_open(W,H)
    cx,cy,R = 300,255,88
    s *= canon_draw_ring(cx,cy; R=R, labels=true, noder=14, route=collect(1:6), name="one RSG output")
    s *= txt(cx, 42, "canonical six-ring anatomy — six discs = six photons total"; size=16, weight="bold")

    labels = [
        (1, "t⁻: fused with the photon delayed from the preceding clock", "#1a9850", 510,390),
        (4, "t⁺: enters a one-clock optical delay, then is destructively fused", "#e08214", 510,95),
        (3, "x⁺ and x⁻: same-clock spatial fusions", "#2166ac", 510,155),
        (2, "y⁺ and y⁻: same-clock spatial fusions", "#7b3294", 510,215),
    ]
    for (_,lab,col,x,y) in labels
        s *= circ(x,y,7,col; stroke="#333")
        s *= txt(x+17,y+4,lab; size=12,fill="#333",anchor="start")
    end
    s *= txt(635,285,"solid hexagon edges = CZ graph inside the resource"; size=12,fill="#555")
    s *= txt(635,308,"outward strokes = waveguide routes, not extra qubits"; size=12,fill="#555")
    s *= txt(635,331,"every colored photon is measured exactly once"; size=12,fill="#555",weight="bold")
    s *= txt(635,354,"in a fusion or a deliberate boundary measurement"; size=12,fill="#555")
    s * "</svg>"
end

canon_patch_pos(i) = begin
    c=(i-1)%3; r=(i-1)÷3
    (150 + 185c + 34r, 145 + 175r)
end

"""The 12 open-grid spatial fusions, in the same row-major ring numbering used by diagram B.

Each tuple is `(id, left_or_upper_ring, its_slot, right_or_lower_ring, its_slot)`.
Keeping this as one table makes the drawn bow ties and the notebook's printed ledger identical.
"""
function canon_spatial_links()
    links = Tuple{String,Int,Int,Int,Int}[]
    nx = 0
    for r in 0:2, c in 0:1
        nx += 1
        a=1+3r+c; b=a+1
        push!(links,("Fₓ$nx",a,3,b,6))       # r_a:x+ ⋈ r_b:x-
    end
    ny = 0
    for r in 0:1, c in 0:2
        ny += 1
        a=1+3r+c; b=a+3
        push!(links,("Fᵧ$ny",a,2,b,5))       # r_a:y+ ⋈ r_b:y-
    end
    links
end

function canon_diagramB(view::Symbol)
    W,H = 1120,650
    colr = view == :dual ? "#c51b7d" : "#2166ac"
    title = view == :hardware ?
        "one 3×3 time slice — the physical spatial-fusion map" :
        (view == :primal ? "one 3×3 time slice — primal-check view" : "the identical slice — dual-check view")
    sub = view == :hardware ?
        "one blue link = one Bell fusion returning both mXX and mZZ; the full check also uses temporal links" :
        (view == :primal ? "same fusion devices; highlight one checkerboard component" : "same fusion devices; highlight the other checkerboard component")
    s = svg_open(W,H)
    # Alternating shaded teaching cells: these are syndrome-graph views, not
    # claims that one ring equals one surface-code data qubit.
    shades = view == :hardware ? Tuple[] :
             (view == :primal ? [(1,2,4,5),(5,6,8,9)] : [(2,3,5,6),(4,5,7,8)])
    for q in shades
        pts=[canon_patch_pos(i) for i in q]
        s *= poly([pts[1],pts[2],pts[4],pts[3]],colr;opacity=0.11)
    end
    # Draw every fusion from the actual photon centres: x+→x- or y+→y-.
    # The short F labels key directly into the complete ledger at right.
    for (fid,a,sa,b,sb) in canon_spatial_links()
        ax,ay=canon_patch_pos(a); bx,by=canon_patch_pos(b)
        p1=leafxy(ax,ay,30,sa;d=9); p2=leafxy(bx,by,30,sb;d=9)
        isx = sa == 3
        s *= fusionlink(p1...,p2...;color=colr,sw=2.0,
                        label=fid,lx=(isx ? 0 : 19),ly=(isx ? -8 : 3))
    end
    for i in 1:9
        x,y=canon_patch_pos(i)
        boundary=Int[]
        c=(i-1)%3; r=(i-1)÷3
        c==0 && push!(boundary,6); c==2 && push!(boundary,3)
        r==0 && push!(boundary,5); r==2 && push!(boundary,2)
        canonroute=setdiff([2,3,5,6],boundary)
        s *= canon_draw_ring(x,y;R=30,labels=true,name="r$i",noder=7,
                             route=canonroute,boundary=boundary)
    end

    # Full endpoint ledger: it removes any ambiguity about what the bow tie consumes.
    tx=690
    s *= "<rect x='655' y='84' width='420' height='445' rx='12' fill='#fafafa' stroke='#c8c8c8' stroke-width='1.4'/>"
    s *= txt(865,112,"the 12 spatial Bell fusions";size=14,weight="bold")
    s *= txt(865,133,"each row names the two measured ring photons";size=10.5,fill="#555")
    for (j,(fid,a,sa,b,sb)) in enumerate(canon_spatial_links())
        y=158+29(j-1)
        axiscol = sa == 3 ? CANON_PORTCOL[3] : CANON_PORTCOL[2]
        s *= txt(tx,y,fid;size=11,fill=axiscol,anchor="start",weight="bold")
        s *= txt(tx+52,y,"r$a:$(CANON_PORTLAB[sa])  ⋈  r$b:$(CANON_PORTLAB[sb])";
                 size=11.5,fill="#333",anchor="start")
    end
    s *= txt(865,522,"bow tie = one destructive Bell measurement";size=10.5,fill="#555")
    s *= txt(W/2,34,title;size=15,weight="bold")
    s *= txt(W/2,55,sub;size=11.5,fill="#555")
    s *= txt(W/2,H-38,"12 same-clock spatial fusions · 12 boundary Z measurements · 9 t⁻ and 9 t⁺ ports are out of plane";size=11,fill="#555")
    footer = view == :hardware ?
        "the syndrome is a mixed parity of spatial and temporal mXX/mZZ records — written explicitly below" :
        "a fusion yields one outcome edge in each view; the views are simultaneous, not alternating rounds"
    s *= txt(W/2,H-18,footer;size=11,fill=colr,weight="bold")
    s * "</svg>"
end

function canon_diagramC()
    W,H=800,900
    s=svg_open(W,H)
    xs=(240,540); ys=(700,455,210); R=43
    for (ti,y) in enumerate(ys)
        for (j,x) in enumerate(xs)
            s *= canon_draw_ring(x,y;R=R,labels=true,name="r$(j), t$(ti)",noder=9,route=collect(1:6))
        end
        # spatial x fusion q3 -> q6
        p1=leafxy(xs[1],y,R,3;d=17); p2=leafxy(xs[2],y,R,6;d=17)
        s *= fusionlink(p1...,p2...;color="#2166ac",label="same-clock fusion",ly=-12)
        s *= txt(W/2,y+48,"both mXX and mZZ enter the classical syndrome graphs";size=11,fill="#555")
    end
    for j in 1:2, ti in 1:2
        x=xs[j]
        p1=leafxy(x,ys[ti],R,4;d=17)
        p2=leafxy(x,ys[ti+1],R,1;d=17)
        s *= fusionlink(p1...,p2...;color="#1a9850",
                        label=(j==2 ? "one-clock delay + temporal fusion" : ""),lx=100,ly=4)
    end
    s *= txt(W/2,36,"space and time in the canonical network";size=16,weight="bold")
    s *= txt(W/2,58,"photons do not persist through a ring: each link terminates in a destructive measurement";size=12,fill="#444")
    s *= txt(W/2,H-20,"time flows upward · green = temporal fusion · blue = same-clock spatial fusion";size=11,fill="#555")
    s * "</svg>"
end

function canon_diagramC2()
    W,H=820,900
    s=svg_open(W,H)
    pos(i,t)=begin
        c=(i-1)%3; r=(i-1)÷3
        (110+150c+46r+38(t-1), 760-225(t-1)-42r)
    end
    for t in 1:3
        # same-slice links
        for r in 0:2,c in 0:1
            a=1+3r+c;b=a+1
            ax,ay=pos(a,t);bx,by=pos(b,t)
            s*=fusionlink(ax,ay,bx,by;color="#2166ac",sw=1.5)
        end
        for r in 0:1,c in 0:2
            a=1+3r+c;b=a+3
            ax,ay=pos(a,t);bx,by=pos(b,t)
            s*=fusionlink(ax,ay,bx,by;color="#7b3294",sw=1.5)
        end
        # temporal links from previous slice
        if t>1
            for i in 1:9
                ax,ay=pos(i,t-1);bx,by=pos(i,t)
                s*=fusionlink(ax,ay,bx,by;color="#1a9850",sw=1.35)
            end
        end
        for i in 1:9
            x,y=pos(i,t)
            s*=canon_draw_ring(x,y;R=16,labels=false,name="",noder=4.8)
            s*=txt(x,y+3,"$i";size=7,fill="#222",weight="bold")
        end
        s*=txt(710,mean([pos(i,t)[2] for i in 1:9]),"slice t=$t";size=12,fill="#333")
    end
    s*=txt(W/2,34,"3×3 rings × three clock slices — a boundary-cut teaching block";size=16,weight="bold")
    s*=txt(W/2,55,"blue/purple = spatial fusions · green = delayed temporal fusions";size=11.5,fill="#555")
    s*=txt(W/2,H-20,"the topological code is carried by check and correlation surfaces through this 3-D complex";size=11,fill="#555")
    s*"</svg>"
end

function canon_diagramD()
    W,H=980,570
    s=svg_open(W,H)
    xs=(125,370,620,855); cy=250
    for i in 1:3
        s*=lin(xs[i]+92,cy,xs[i+1]-92,cy;stroke="#777",sw=2)
        s*="<polygon points='$(xs[i+1]-92),$cy $(xs[i+1]-106),$(cy-7) $(xs[i+1]-106),$(cy+7)' fill='#777'/>"
    end
    s*=canon_draw_ring(xs[1],cy;R=55,labels=true,noder=10,route=collect(1:6),name="")
    s*=txt(xs[1],80,"① SOURCE";size=15,weight="bold")
    s*=txt(xs[1],103,"one six-photon ring";size=12,fill="#555")

    s*=canon_draw_ring(xs[2],cy;R=55,labels=true,noder=10,route=collect(1:6),name="")
    s*=txt(xs[2],80,"② ROUTE";size=15,weight="bold")
    s*=txt(xs[2],103,"four spatial · t⁻ incoming · t⁺ delayed";size=11.5,fill="#555")
    s*=txt(xs[2],395,"waveguide length schedules coincidence";size=11,fill="#555")

    # Fusion panel: six bow ties, no surviving node.
    for k in 1:6
        a=2pi*(k-1)/6
        x=xs[3]+62cos(a);y=cy+62sin(a)
        s*=circ(x,y,8,CANON_PORTCOL[k];stroke="#333")
        s*=txt(x,y+3.5,"⋈";size=12,fill="white",weight="bold")
    end
    s*=txt(xs[3],80,"③ MEASURE";size=15,weight="bold")
    s*=txt(xs[3],103,"six photons consumed in six pairings";size=11.5,fill="#555")
    s*=txt(xs[3],395,"clicks → (mXX, mZZ, success/loss flags)";size=11,fill="#555")

    s*="<rect x='778' y='190' width='154' height='120' rx='12' fill='#fff4df' stroke='#c47a16' stroke-width='2'/>"
    s*=txt(xs[4],220,"CLASSICAL";size=14,weight="bold",fill="#a85f08")
    s*=txt(xs[4],247,"check parities";size=12,fill="#333")
    s*=txt(xs[4],270,"streaming decoder";size=12,fill="#333")
    s*=txt(xs[4],293,"Pauli frame / bases";size=12,fill="#333")
    s*=txt(xs[4],80,"④ RECORD";size=15,weight="bold")
    s*=txt(xs[4],103,"no quantum photon survives";size=11.5,fill="#555")

    s*=txt(W/2,32,"life of one canonical ring — emit, route, destructively fuse, decode";size=17,weight="bold")
    s*=txt(W/2,500,"At a planar boundary a selected fusion is replaced by a single-photon measurement, and an entirely disconnected resource may be omitted.";size=11.5,fill="#555")
    s*=txt(W/2,524,"There is no pad/retire step and no hidden leaf layer: the source emitted exactly the six photons drawn in panel ①.";size=11.5,fill="#2166ac",weight="bold")
    s*"</svg>"
end
