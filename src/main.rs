use std::collections::HashMap;

use image::ImageBuffer;
use kiddo::{float::distance::squared_euclidean, KdTree};
use rand::seq::SliceRandom;
use toodee::{TooDee, TooDeeOps};

fn gen_colors() -> (Vec<image::Rgb<u8>>, Vec<[f32; 3]>) {
    let mut rgbs = vec![];
    for r in 0..=255 {
        for g in 0..=255 {
            for b in 0..=255 {
                rgbs.push(image::Rgb([r, g, b]));
            }
        }
    }

    rgbs.shuffle(&mut rand::thread_rng());

    let mut oklabs = vec![];
    for rgb in &rgbs {
        let oklab = oklab::srgb_to_oklab(oklab::RGB::new(rgb[0], rgb[1], rgb[2]));
        oklabs.push([oklab.l, oklab.a, oklab.b]);
    }

    (rgbs, oklabs)
}

fn coord_to_int(x: usize, y: usize) -> usize {
    (x << 12) | y
}

fn int_to_coord(i: usize) -> (usize, usize) {
    (i >> 12, i & ((1 << 12) - 1))
}

const NEIGHBORS: &[(isize, isize)] = &[
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
];

fn neighbors<T: Copy>(
    x: usize,
    y: usize,
    data: &TooDee<T>,
) -> impl Iterator<Item = (usize, usize, T)> + '_ {
    NEIGHBORS
        .iter()
        .map(move |&(dx, dy)| (x as isize + dx, y as isize + dy))
        .filter(|&(new_x, new_y)| {
            new_x >= 0
                && new_x < data.num_rows() as isize
                && new_y >= 0
                && new_y < data.num_cols() as isize
        })
        .map(|(new_x, new_y)| (new_x as usize, new_y as usize))
        .map(|(new_x, new_y)| (new_x, new_y, data[new_x][new_y]))
}

fn empty_neighbors<T: Copy>(
    x: usize,
    y: usize,
    data: &TooDee<Option<T>>,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    neighbors(x, y, data).filter_map(|(x, y, ref t)| if t.is_none() { Some((x, y)) } else { None })
}

fn target_color(x: usize, y: usize, data: &TooDee<Option<usize>>, colors: &[[f32; 3]]) -> [f32; 3] {
    neighbors(x, y, data)
        .flat_map(|(_, _, color_idx)| color_idx.map(|c| (1.0, colors[c])))
        .reduce(|(count, acc), (_, e)| (count + 1.0, [acc[0] + e[0], acc[1] + e[1], acc[2] + e[2]]))
        .map(|(c, acc)| [acc[0] / c, acc[1] / c, acc[2] / c])
        .unwrap()
}

fn main() {
    let (rgbs, oklabs) = gen_colors();
    let mut kdtree: KdTree<_, 3> = KdTree::new();
    let mut available: HashMap<(usize, usize), [f32; 3]> = HashMap::new();
    let mut buf: TooDee<Option<usize>> = TooDee::new(1 << 12, 1 << 12);

    for (color_idx, lab) in oklabs.iter().enumerate() {
        let (x, y) = if color_idx == 0 {
            (buf.num_rows() / 2, buf.num_cols() / 2)
        } else {
            int_to_coord(kdtree.nearest_one(lab, &squared_euclidean).1)
        };
        if let Some(old) = available.remove(&(x, y)) {
            kdtree.remove(&old, coord_to_int(x, y));
        }
        buf[x][y] = Some(color_idx);

        for (nx, ny) in empty_neighbors(x, y, &buf) {
            if let Some(old) = available.insert((nx, ny), target_color(nx, ny, &buf, &oklabs)) {
                kdtree.remove(&old, coord_to_int(nx, ny));
            }
            kdtree.add(&available[&(nx, ny)], coord_to_int(nx, ny));
        }
    }

    let imgbuf = ImageBuffer::from_fn(1 << 12, 1 << 12, |x, y| {
        rgbs[buf[x as usize][y as usize].unwrap()]
    });

    imgbuf.save("out.png").unwrap();

    let mut all_colors: Vec<usize> = buf.into_iter().flatten().collect();
    assert_eq!(all_colors.len(), 1 << 24);
    all_colors.sort();
    all_colors.iter().enumerate().for_each(|(i, x)| {
        assert_eq!(i, *x);
    });
}
