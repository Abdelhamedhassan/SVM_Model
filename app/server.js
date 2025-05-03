const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = 8081;


const mimeTypes = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
};

const server = http.createServer((req, res) => {

  let filePath = req.url;


  if (filePath === "/") {
    filePath = "/template/index.html";
  }


  if (filePath === "/prediction") {
    filePath = "/template/prediction.html";
  } else if (filePath === "/dashboard") {
    filePath = "/template/dashboard.html";
  }


  const extname = path.extname(filePath);
  const contentType = mimeTypes[extname] || "application/octet-stream";


  const fullPath = path.join(__dirname, "..", filePath.replace(/^\//, ""));


  fs.readFile(fullPath, (err, content) => {
    if (err) {
      if (err.code === "ENOENT") {
        // File not found
        res.writeHead(404);
        res.end("File not found");
      } else {
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
      }
    } else {
      res.writeHead(200, { "Content-Type": contentType });
      res.end(content, "utf-8");
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
});
