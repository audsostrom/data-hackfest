#!/usr/bin/env node

const fs = require("fs");
fs.createReadStream("./movies.csv")
  .on("data", (chunk) => {
    console.log(chunk);
  })
  .on("error", (error) => {
    console.log(error);
  });