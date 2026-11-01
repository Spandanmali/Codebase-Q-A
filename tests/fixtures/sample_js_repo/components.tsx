import React, { useState as state } from "react";
import { helper } from "./util.js";
import "./side-effect";
export { helper as exportedHelper } from "./util.js";

export default function App() {
  const local = () => helper();
  return <div>{state(local)}</div>;
}

class Service {
  run() {
    return helper();
  }
}

const Widget = () => <Service />;
const loaded = require("./nested");
const external = require("express");
