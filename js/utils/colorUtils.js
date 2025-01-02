/**
 * Convert HSL color values to RGB
 * @param {number} h - Hue (0-1)
 * @param {number} s - Saturation (0-1)
 * @param {number} l - Lightness (0-1)
 * @returns {number[]} Array of [r, g, b] values (0-255)
 */
export function hslToRgb(h, s, l) {
    let r, g, b;

    if (s === 0) {
        r = g = b = l; // Achromatic
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }

    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Convert RGB values to hexadecimal color string
 * @param {number} r - Red (0-255)
 * @param {number} g - Green (0-255)
 * @param {number} b - Blue (0-255)
 * @returns {string} Hex color string
 */
export function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/**
 * Convert HSL values directly to hex color string
 * @param {number} h - Hue (0-1)
 * @param {number} s - Saturation (0-1)
 * @param {number} l - Lightness (0-1)
 * @returns {string} Hex color string
 */
export function hslToHex(h, s, l) {
    const [r, g, b] = hslToRgb(h, s, l);
    return rgbToHex(r, g, b);
}

/**
 * Adjust the shade of a hex color
 * @param {string} hex - Hex color string
 * @param {number} amount - Amount to adjust (-1 to 1)
 * @returns {string} Modified hex color
 */
export function shadeHexColor(hex, amount = -0.2) {
    hex = hex.replace('#', '');
    
    const r = Math.max(0, Math.min(255, parseInt(hex.slice(0, 2), 16) + amount * 100));
    const g = Math.max(0, Math.min(255, parseInt(hex.slice(2, 4), 16) + amount * 100));
    const b = Math.max(0, Math.min(255, parseInt(hex.slice(4, 6), 16) + amount * 100));

    return rgbToHex(r, g, b);
}

/**
 * Get contrasting text color (black or white) for a background color
 * @param {string} hexColor - Hex color string
 * @returns {string} '#000000' for light backgrounds, '#FFFFFF' for dark
 */
export function getContrastColor(hexColor) {
    hexColor = hexColor.replace('#', '');
    
    const r = parseInt(hexColor.slice(0, 2), 16) / 255;
    const g = parseInt(hexColor.slice(2, 4), 16) / 255;
    const b = parseInt(hexColor.slice(4, 6), 16) / 255;

    // Calculate relative luminance
    const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    return L > 0.179 ? "#000000" : "#FFFFFF";
} 