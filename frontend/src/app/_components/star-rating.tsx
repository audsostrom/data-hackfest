"use client"
import { useMutation } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { api } from "~/trpc/react";
import { Rating } from 'react-simple-star-rating'

export default function StarRating() {
  const [rating, setRating] = useState(0);

  const handleRating = (rate: number) => {
    setRating(rate)
    // createRating.mutate({rate});
  }

  const createRating = api.post.create.useMutation();

  return(
    <Rating
    onClick={handleRating}
    initialValue={rating}
    />
  )

  //   userId: ctx.session.user.id,
  // movieId: input.movieId,
  // rating: input.rating,
  // desc: input.desc,
}

