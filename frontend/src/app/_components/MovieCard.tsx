import {Movie} from "~/server/db/schema";
import {Card, Paper} from "@mui/material";
import Typography from "@mui/material/Typography";
import Image from "next/image";
import Box from "@mui/material/Box";

export interface MovieCardProps {
  movie: Movie;
}


export default function MovieCard({movie}: MovieCardProps){
  return(
    <Paper sx={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'space-between',
      p: 2,
      borderRadius: 2,
      overflow: 'hidden',
      boxShadow: 3,
      width: {
        xs: '100%',
        md: '95%',
      },
      height: '100%',
      gap: 2,
    }}>
      <Typography variant={'body2'} sx={{
        fontWeight: 'bold',
        textAlign: 'center',
        fontSize: '0.8rem',
      }}>
        {movie.title}
      </Typography>
      <Box sx={{
        height: 100,
        width: '100%',
        maxWidth: '300px',
        overflow: 'hidden',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}>
        <img
          src={'https://m.media-amazon.com/images/M/MV5BYzI2YmZhZTQtMWQ1OC00YTMxLWJlNzgtYjYyZGZjOWEyNmQxXkEyXkFqcGdeQWFybm8@._V1_.jpg'}
          alt={'movie banner'}/>
      </Box>
    </Paper>
  );
}