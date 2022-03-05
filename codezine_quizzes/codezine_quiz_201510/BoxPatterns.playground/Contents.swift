struct Point {
    var x: Int
    var y: Int
}

let AbsoluteOrigin = Point(x: 0, y: 0)

func ==(lhs: Point, rhs: Point) -> Bool {
    return lhs.x == rhs.x && lhs.y == rhs.y
}

struct Tile: Equatable {
    var width: Int
    var height: Int
}

func ==(lhs: Tile, rhs: Tile) -> Bool {
    return lhs.width == rhs.width && lhs.height == rhs.height
}

let Landscape = Tile(width: 3, height: 2)
let Portrait = Tile(width: 2, height: 3)

enum Filler: Character {
    case UpperLeft = "┏"
    case UpperRight = "┓"
    case LowerLeft = "┗"
    case LowerRight = "┛"
    case VerticalBar = "┃"
    case HorizontalBar = "━"
    case Middle = "　"
    case None = "Ｎ"
}

struct Cell: CustomStringConvertible {
    var filled = false
    var filler: Filler = .None
    
    func isFilled() -> Bool {
        return filled
    }
    
    mutating func fill(filler: Filler) {
        self.filled = true
        self.filler = filler
    }
    
    mutating func empty() {
        self.filled = false
        self.filler = .None
    }
    
    func isUpperLeft() -> Bool {
        return filler == .UpperLeft
    }
    
    var description: String {
        return String(filler.rawValue)
    }
}

struct Plane: CustomStringConvertible {
    var area: [[Cell]]
    
    init(width: Int, height: Int) {
        area = [[Cell]](count: height, repeatedValue: [Cell](count: width, repeatedValue: Cell()))
    }
    
    mutating func place(origin: Point, tile: Tile) -> Bool {
        var result = area
        for var i = origin.y; i < origin.y + tile.height; ++i {
            if i >= result.count {
                return false
            }
            for var j = origin.x; j < origin.x + tile.width; ++j {
                if j >= result[i].count {
                    return false
                }
                if result[i][j].isFilled() {
                    return false
                }
                
                let filler: Filler
                if i == origin.y { // upper
                    if j == origin.x { // left
                        filler = .UpperLeft
                    } else if j == origin.x + tile.width - 1 { // right
                        filler = .UpperRight
                    } else {
                        filler = .HorizontalBar
                    }
                } else if i == origin.y + tile.height - 1 { // lower
                    if j == origin.x { // left
                        filler = .LowerLeft
                    } else if j == origin.x + tile.width - 1 { // right
                        filler = .LowerRight
                    } else {
                        filler = .HorizontalBar
                    }
                } else { // middle high
                    if j == origin.x { // left
                        filler = .VerticalBar
                    } else if j == origin.x + tile.width - 1 { // right
                        filler = .VerticalBar
                    } else {
                        filler = .Middle
                    }
                }
                result[i][j].fill(filler)
            }
        }
        
        area = result
        return true
    }
    
    func isComplete() -> Bool {
        for var i = area.count - 1; i >= 0; --i {
            for var j = area[i].count - 1; j >= 0; --j {
                if !area[i][j].isFilled() {
                    return false
                }
            }
        }
        return true
    }
    
    func findNextOrigin() -> Point? {
        for var i = 0; i < area.count; ++i {
            for var j = 0; j < area[i].count; ++j {
                if !area[i][j].isFilled() {
                    return Point(x: j, y: i)
                }
            }
        }
        return nil
    }
    
    // returns the last point at which a tile is validly placed
    func findLastOrigin(smallerThan: Point) -> Point? {
        if smallerThan == AbsoluteOrigin {
            return nil
        }
        if area[0].count <= smallerThan.x || area.count <= smallerThan.y {
            return nil
        }
        
        let i = smallerThan.y
        for var j = smallerThan.x - 1; j >= 0; --j {
            if area[i][j].isUpperLeft() {
                return Point(x: j, y: i)
            }
        }
        
        for var i = smallerThan.y - 1; i >= 0; --i {
            for var j = area[i].count - 1; j >= 0; --j {
                if area[i][j].isUpperLeft() {
                    return Point(x: j, y: i)
                }
            }
        }
        
        return nil
    }
    
    func isPlaced(origin: Point, tile: Tile) -> Bool {
        for var i = origin.y; i < origin.y + tile.height; ++i {
            if i >= area.count {
                return false
            }
            for var j = origin.x; j < origin.x + tile.width; ++j {
                if j >= area[i].count {
                    return false
                }
                if !area[i][j].isFilled() {
                    return false
                }
            }
        }
        return true
    }
    
    mutating func reset() {
        displace(AbsoluteOrigin, tile: Tile(width: area[0].count, height: area.count))
    }
    
    mutating func displace(origin: Point, tile: Tile) -> Bool {
        var result = area
        for var i = origin.y; i < origin.y + tile.height; ++i {
            if i >= result.count {
                return false
            }
            for var j = origin.x; j < origin.x + tile.width; ++j {
                if j >= result[i].count {
                    return false
                }
                if !result[i][j].isFilled() {
                    return false
                }
                
                result[i][j].empty()
            }
        }
        
        area = result
        return true
    }
    
    mutating func rewind(origin: Point, tile: Tile) -> Point {
        var result = origin
        while let o = findLastOrigin(result) {
            if isPlaced(o, tile: tile) {
                if displace(o, tile: tile) {
                    result = o
                    continue
                }
            }
            break
        }
        return result
    }
    
    mutating func startAt(origin: Point, tile: Tile) -> Int {
//        print(self)
        if place(origin, tile: tile) {
            if isComplete() {
                print("Found")
                print(self)
                displace(origin, tile: tile)
                if let o = findLastOrigin(rewind(origin, tile: Portrait)) {
                    displace(o, tile: Landscape)
                    return 1 + startAt(o, tile: Portrait)
                }
//                print("DONE!")
                return 1
            }
            if let o = findNextOrigin() {
                return startAt(o, tile: Landscape)
            }
//            print("no next origin: \(origin), \(tile)")
            return 0
        }
        
        if tile == Landscape {
            return startAt(origin, tile: Portrait)
        }
        
        if let o = findLastOrigin(rewind(origin, tile: Portrait)) {
            displace(o, tile: Landscape)
            return startAt(o, tile: Portrait)
        }
        
//        print("rewound all: \(origin), \(tile)")
        return 0
    }
    
    var description: String {
        var s = ""
        for var i = 0; i < area.count; ++i {
            for var j = 0; j < area[i].count; ++j {
                s += area[i][j].description
            }
            s += "\n"
        }
        return s
    }
}

func calculateSize(tiles: Int) -> Int {
    return 6 * tiles
}

func calculateMaxHeight(size: Int) -> Int {
    for var i = 1, max = i; i < size; ++i {
        if i * i > size {
            return max
        }
        if size % i == 0 {
            max = i
        }
    }
    return 1
}

func countPatterns(size: Int) -> Int {
    var total = 0
    for var h = 2, max = calculateMaxHeight(size); h <= max; ++h {
        let w = size / h
        var plane = Plane(width: w, height: h)
        let n = plane.startAt(AbsoluteOrigin, tile: Landscape)
        total += (h == w) ? n / 2 : n
    }
    
    total *= 2
    
    return total
}

func get1stInt(defaultValue: Int) -> Int {
    var n = defaultValue
    if Process.argc > 1 {
        if let i = Int(Process.arguments[1]) {
            if i > 0 {
                n = i
            }
        }
    }
    return n
}

// http://codezine.jp/article/detail/8985
// $ swift -O this.swift 20
// => number of patterns: 610
func main() {
    let tiles = get1stInt(1)
    let size = calculateSize(tiles)
    let total = countPatterns(size)
    print("number of patterns: \(total)")
}

main()
