"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { api } from "~/trpc/react";
import styles from "../index.module.css";



import * as React from 'react';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import { Movie } from "~/server/db/schema";
import { number } from "zod";

//** typescript prop pain */
interface IProps {
    movies: Movie[]
  }
  
//** example movie selection dropdown */
export function SelectMovie({movies}: IProps) {
  const [movieNum, setMovieNum] = useState('');

  // the useQuery hook allows api calls to contain state
  // check out https://tanstack.com/query/v5/docs/framework/react/guides/queries
  // for the full list of states
  const { data: selectedMovie, isLoading: isGetting } =
  api.movie.getMovie.useQuery({id: Number(movieNum)})

  const handleChange = (event: SelectChangeEvent) => {
    setMovieNum(event.target.value as string);
  };

  return (
    <>
    <FormControl fullWidth>
    <InputLabel id="demo-simple-select-label">Movie</InputLabel>
    <Select
      labelId="demo-simple-select-label"
      id="demo-simple-select"
      value={movieNum}
      label="Movie"
      onChange={handleChange}
    >
      {movies.map((movie)=><MenuItem key={movie.id} value={movie.id}>{movie.title}</MenuItem>)}
    </Select>
  </FormControl>
  {/* needs to have an optional symbol (?) since selectedMovie can be generalized to null */}
  <p>Selected {selectedMovie?.title}</p>
  </>
  );
}
